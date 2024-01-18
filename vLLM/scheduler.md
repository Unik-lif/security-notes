## 概览
在这一篇博客中我们集中精力尝试分析`vLLM`项目中的`swap`机制，这个主要是`scheduler`来干，意思为在`cpu_blocks`与`gpu_blocks`之间进行交换的过程。

## 分析
`swap`过程发生在`LLM Engine`调用`run_engine`之后，利用`schedule`函数对`scheduler`调度器内存放的三个请求队列的操作，这三个队列分别为：
1. `running`队列：表示马上就要去处理的请求
2. `swapped`队列：表示一度放在`running`队列中，希望马上去处理的请求，但这一部分请求因为资源不足被其他请求所抢占，而被迫从`running`队列中弹出进入`swapped`队列，表示要把当前的一些`gpu blocks`放到`cpu blocks`中进行暂存
3. `waiting`队列：表示之后需要拿去处理的请求，在进行`add_request`操作之后，所有的请求默认先放在这边，之后适时向前两个拥有更高优先级的请求做发射


一个请求进入后的状态变化关系图：
```
request processing route: 

              allocate: allocate gpu physical               _append_slot: logical_blocks?
              blocks due to logical_token_blocks length     generate new block according to copy-on-write
waiting List -------------------------------> running List -------->
                                                   |   ^
                                         swap out  |   |
                        swapped List  <-------------   |
                              |      swap in           |
                              |------------------------>
                                  
```
看的出来，块的分配遵循两个步骤，一阶段是针对`seq_group`中的`seqs`分配`logical_blocks`，只分配一组，`beam search`中对应的其他`seq`会通过浅拷贝的方式来接上（浅拷贝在`rust`和`c++`中很常用）。

二阶段可以理解成`CD`好了开大招`R`，会根据最后一个`block`的引用次数来决定替换最后一块的情况。当然，作为反映输入`input`的块它的引用次数一直都会比较高，所以`_append_slot`可以算是程序真正跑起来之后会发生的事情。在`schedule`的第一次调用时，所有的请求，即`seq_groups`都会处于`waiting list`之中，这一次调用后的结局是，他们只会塞到`running list`中，但是还没有经历到第二阶段。

特别需要注意到是，`running list`中存在两种类型的`seq_groups`，分别是一阶段和二阶段。
### _schedule函数
在函数`_schedule`中，反映了对于`swapped`块做的一些操作：
```python
    def _schedule(self) -> Tuple[SchedulerOutputs, List[str]]:
        # Blocks that need to be swaped or copied before model execution.
        # 下面这一部分字典将会如实记录，因为gpu显存限制，哪些块需要被swap_out，或者说因为gpu显存充足，哪些先前被swap到cpu memory的块可以在现在swap_in了
        # copy则反映接下来哪些块可以马上放到gpu上跑了
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()

        # NOTE(woosuk): We prioritize the sequence groups in the RUNNING state
        # in order to minimize the preemption overheads.
        # Preemption happens only when there is no available slot to keep all
        # the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        # 简单的对running队列做先来后到排序
        self.running = self.policy.sort_by_priority(now, self.running)
```
调度顺序存在上面所述的优先级，现在我们先看最高优先级，对于`running`队列的操作如下所示：
```python
        # Reserve new token slots for the running sequence groups
        # 每一个SequenceGroup都对应一个Request请求
        # running队列会从一开始记录调度情况，最终会成为经过本次调度之后的running队列信息
        # preempted队列记录被抢占了的请求
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        # 如果running队列有请求的话
        while self.running:
            # 先弹出第一个最高优先级的，我们接下来要对它做处理了
            seq_group = self.running.pop(0)
            # 在block_manager之中，简单判别，当前gpu_allocator中没有足够的blocks以让每个seq都能找到一个block生存的话
            # 抢占人家是不可避免的了
            while not self.block_manager.can_append_slot(seq_group):
                # 如果除了第一个元素，还有其他元素在running list之中
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    # 抢占低优先级的，让低优先级的seq_group把第一步吃进去的gpu_block吐出来先
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    # 被抢占的队列记录victim_seq_group
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    # 没有别人可以欺负，只好自己先吐出来第一步吃进去的gpu_block，静候转机
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                # 如果gpu_allocator有充足的blocks让seq_group进入二阶段
                # 那么直接通过_append_slot分配给其二阶段所需要的资源，并在running数组中加上seq_group
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running
```
需要注意的是，`append_slot`和`allocate`的操作方法和行为逻辑存在不同，我们之后做区分。大体而言，在`waiting List`向其他阵列弹射时，不可避免地需要进行`_allocate`操作，然后再考虑走`_append_slot`这条路，在这时候才正式算是准备拿去处理数据了。这两个步骤分别都会抢占一部分`Blocks`资源，因此在必要时刻需要把部分`seq_group`暂存到`GPU`之中，以腾出足够资源给高优先级的`seq_group`使用。

对应的资源抢占函数`_preempt`如下所示：由于默认情况下的`preemption_mode`没有东西，`SWAP`状态是避免不了的了。因此该函数简单总结就是把指定的`seq_group`定性为将要被`SWAP`的东东，之后通过`_preempt_by_swap`来做具体处理。
```python
    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, 'Invalid preemption mode.'
```
该函数结构如下所示：通过`_swap_out`将`seq_group`中状态为`RUNNING`（既然先前从`running list`中被抢占出来，这个状态自然也为`RUNNING`）的`seq`，在给他们打上`SWAPPED`状态标记之后，将部分信息存放到`blocks_to_swap_out`字典中。
```python
    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)
```
具体来说，`_swap_out`函数如下所示：需要事先判断`seq_group`所对应的`physical blocks`量是否是能够被`block_manager`所能够承接的，由于是`swapped`状态，比较是与`cpu_allocator`相进行。在`block_manager`中进行了真正的`swap_out`操作之后，设置`seq`状态为`SWAPPED`，这个函数便结束了。
```python
    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
```
接下来我们研究函数`swap_out`，如下所示：该函数实现了从`GPU`块向`CPU`块的交换，并把最后的`mapping`结果返回出来，以在之后`swap_in`时能够重新恢复状态。
```python
    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            # 一般情况下finished不会出现，需要主动通过abort或者stop来设置
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            # 找到seq所对应的block_table，这一步似乎是allocate所完成
            block_table = self.block_tables[seq.seq_id]
            # 将该seq所对应的block_table写一个mapping记录下来，如果mapping已经在cpu_allocator中存在，
            # 则增大ref_count
            # 
            # 否则需要额外分配一下这个cpu_block
            # 此外要把gpu_block丢掉
            #
            # 此时的block_table从gpu_block_table转化为了cpu_block_table，存放cpu_block号
            # 在以后swap_in的时候，可以通过mapping将cpu_block_table转化为gpu_block_table
            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping
```
现在我们看向函数`_schedule`的第二部分，对于`swapped`队列的处理。
```python
        # Swap in the sequence groups in the SWAPPED state if possible.
        # 对swapped队列进行排序
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        # 如果swapped队列中有值，并且此时blocks_to_swap_out中没有值
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            # can_swap_in => gpu_allocator free.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = len(self.running)
            # 看样子这里是有bug的
            # 原因也很简单，类型搞错了，前面的是一个seq_group中的seqs数目
            # 后面是seq_groups总数
            if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )
```
`swapped`进入二阶段的优先级比较低，如果先前`blocks_to_swap_out`中已经有了项，那么就不进行`swap`处理，这个也比较好理解，如果`blocks_to_swap_out`中有值，说明`gpu_allocator`中的空余块并不足够，在这种情况下还想着`swap_in`去使用`gpu_allocator`中的资源，只会造成资源更大的缺口。

剩下的感觉比较好理解，`swapp_in`还会更新`seq`状态，之后就可以通过`_append_slot`开二阶段，塞到`running list`之中。

之后维护了一个`num_batched_tokens`，现在这个队列中可以视作全部都开了二阶段的`seq_groups`构成的批处理块，记录了这边`seqs`数目的总和。不过根据后面的观察，似乎这边也有一个`bug`，`num_batched_tokens`记录了`seqs`数目，而不是`tokens`数目。

在`swapped`队列被处理好后，我们看向`waiting`队列。这个队列必须要在`swapped`队列之中彻底没有值了以后，才会被拿来处理。在`schedule`刚开始的阶段，这种情景是适用的。
```python
        # Join waiting sequences if possible.
        # prompt队列被用于记录从waiting list之中弹出，提高了优先级而被放入swapped和running队列中的seq_group
        prompt_group_ids: List[str] = []
        # NOTE(woosuk): The sequence groups in the SWAPPED state are strictly
        # prioritized over the sequence groups in the WAITING state.
        # This is because we want to bound the amount of CPU memory taken by
        # the swapped sequence groups.
        if not self.swapped:
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]
                # If the sequence group has been preempted in this step, stop.
                # 感觉似乎没有这种可能性？
                if seq_group in preempted:
                    break
                # If the sequence group cannot be allocated, stop.
                # 没有足够空间进行allocate让waiting list中的seq_group进入一阶段
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                # 一次只能处理max_num_batched_tokens这么多的tokens
                # 这边似乎也有一个存疑的bug，或许我对num_batched_tokens的理解存在偏差
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if (num_batched_tokens + num_prompt_tokens
                    > self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                # 先前提到的bug在这边重复出现了
                num_new_seqs = seq_group.num_seqs(status=SequenceStatus.WAITING)
                num_curr_seqs = len(self.running)
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                seq_group = self.waiting.pop(0)
                # 开启一阶段，但是还没有开启二阶段
                # 同样是running，有些running开启了一阶段，有些running则开启了二阶段
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                prompt_group_ids.append(seq_group.request_id)

        scheduler_outputs = SchedulerOutputs(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        if not self.log_stats:
            return scheduler_outputs, prompt_group_ids
```
既然看到这边了，我们应该仔细来考虑一下一阶段和二阶段之间的不同之处了：

首先一阶段对应着`_allocate`函数，该函数似乎对`seq_group`只进行一次分配：也就是一个`seq_group`对应一次`allocate`分配。
```python
    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING
```
对于`allocate`函数做分析。
```python
    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same prompt.
        # 思路可能还是Copy on Write，虽然一个seq_group中的seqs是通过tokenizer处理后
        # 有可能有所不同的seq，但是可能还是存在较大的重合部分
        # 这边通过采样，选取第一个seq
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        # 总之为每个logcial_token_block都分配一个block与之进行对应
        # 初始化的时候，默认每个seq所产生的block都与第一个一样，所以在这边会把ref_count一并记作beam search所对应的top-k个
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            # 记录gpu_allocator所指定的那些block
            block_table.append(block)

        # Assign the block table for each sequence.
        # 在block_tables中设置seq所对应的block_table均为同一组
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()
```
对于二阶段的`_append_slot`函数做分析：这个函数对于`running`状态的`seq`做操作
```python
    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                # last_block_number, new_block_number
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]
```
对应的函数为`append_slot`，如果尾`token`被多次`ref`，则替换其为独立的一个。
```python
    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        # block_table基于之前的allocate，因此append_slot被我称为二阶段
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]
        # 我感觉一般这种情况不会出现，先前block_table分配空间的时候，就是按照logical_blocks数目来的
        # 或者说，在跑起来之后才有可能出现
        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

```

### 一些问题：
不过对于这两个`allocate`以及`append_slot`为什么要这么做，以及`blocks_to_copy`的相关细节还是缺乏了解，可能得重新看一下文献，了解一下这边所说的`copy-on-write`机制。

感觉`append_slot`不管从哪个方向去看都非常诘屈聱牙？

刚刚看了一眼论文，似乎可以理解了。

`seq_group`中存放的`seqs`都是一样的，作为`input`来看待，在跑`append_slot`之前，`beam search`现在其实还没有开始，只不过先给它预留了一些块。现在往下生成时，会通过`copy-on-write`方式向下生成，所以是从最后一位来看，看其的`ref_count`是否为`1`，然后对应的需要拷贝的块会存放到`blocks_to_copy`之中（其实它先前的块可能也需要拷贝，但是在这边似乎还没体现出来，可能在下一个步骤中出现？）

先前的块复制是浅拷贝，从这里可以看出`cop-on-write`的思想，不过由于东西还没完全跑起来看，我们需要赶紧去看`schedule`之后的步骤了。

### schedule函数
上面对`_schedule`函数做了较为详细的分析，现在我们看看外面的函数：
```python
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.

        # _schedule函数返回了由blocks_to_swap_in/out, blocks_to_copy组成的调度结果
        # 以及从waiting list弹出来的seq_group信息
        scheduler_outputs, prompt_group_ids = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in self.running:
            is_prompt = seq_group.request_id in prompt_group_ids

            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            # 总之似乎遍历了running list中seq_group中全部seq
            # 存储了他们的seq_data与对应的block_tables信息
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
            # 根据上面那么多的东西包起来了一个seq_group_metadata数据
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                # 这个seq_group是否涉及从waiting list弹出请求进入到更高优先级之中？
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs
```
看起来他的行为还是比较简单的，我们再看看这个函数返回到了哪里去？这一则博客到这边也就结束了。