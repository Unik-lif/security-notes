## 概览
在这一篇博客中我们集中精力尝试分析`vLLM`项目中的`swap`机制，意思为在`cpu_blocks`与`gpu_blocks`之间进行交换的过程。

## 分析
`swap`过程发生在`LLM Engine`调用`run_engine`之后，利用`schedule`函数对`scheduler`调度器内存放的三个请求队列的操作，这三个队列分别为：
1. `running`队列：表示马上就要去处理的请求
2. `swapped`队列：表示一度放在`running`队列中，希望马上去处理的请求，但这一部分请求因为资源不足被其他请求所抢占，而被迫从`running`队列中弹出进入`swapped`队列，表示要把当前的一些`gpu blocks`放到`cpu blocks`中进行暂存
3. `waiting`队列：表示之后需要拿去处理的请求，在进行`add_request`操作之后，所有的请求默认先放在这边，之后适时向前两个拥有更高优先级的请求做发射


有一个东西似乎还没有搞清楚。。。即logical_blocks？

先歇一会儿，做一会儿Lab吧。

一个请求进入后的状态变化关系图：
```
request processing route: 

              allocate: allocate gpu physical               _append_slot: logical_blocks?
              blocks due to logical_token_blocks length     
waiting List -------------------------------> running List -------->
                                                   |   ^
                                         swap out  |   |
                        swapped List  <-------------   |
                              |      swap in           |
                              |------------------------>
                                  
```
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