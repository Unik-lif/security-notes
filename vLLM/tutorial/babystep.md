## 代码主干分析
我们主要强调针对 prefix 这一明显利用的方式来对 vllm 的框架做简单的分析

我们使用的大模型类型是 llama2 ，调研使用的程序是自带的 offline_inference_with_prefix.py

程序主干：
- 利用 LLM 来配置对应大模型的参数和应用框架（√）
- 利用 genrate 来生成我们想要的结果（需要进一步分析）

在利用 prefix 的情况下， generating_prompts 是我们的 prefix + prompt 后得到的长字符串结果

在 generate 函数中，将全部的 request 通过 _add_request 逐个加入需要进行解析的队列之中，加入完之后，通过 _run_engine 来运行程序
### LLM 初始化

### _add_request
把每个 prompt 和 request_id 值对应在一起，再调用 llm_engine 中的 add_request 函数来运转。

首先，记录 arrival_time ，即请求所到来的时间，如果 prompt_token_ids 为空，则通过 encode_request 来生成，最后这个行为其实是调用了 transformer 库中的 encode 行为。最终通过 encode_request 使得我们可以得到 prompt_token_ids 的信息，并且进一步储存起来。

在获取 prompt_token_ids 之后，结合 seq_id 和 block_size 等信息之后，尝试生成了 Sequence 语句。

Sequence 语句的生成设计重要的针对 block 的使用，我们需要仔细研究一下：

Sequence 特殊事项如下：
- 每个 Sequence 实际上与每一条请求语句对应，拥有 seq_id 和当前的状态 status ，对于 prompt 信息，生成了 prompt_token_ids 的数据，以 SequenceData 的方式进行包装，作为其所存在的数据 data
- prompt_token_ids 的信息被函数 `_append_tokens_to_blocks` 使用
- sequence 中的 logic_token_blocks 记录了 prompt_tokens_ids 会存放到的 blocks 地址
- 每个 SequenceGroup 实际上就是只有一个 Sequence ，这一点容易忽视掉

在对 Sequence 做了封装处理后，通过 add_or_get_prefix 尝试利用 prefix 以加快速度。 prefix_pos 记录了在 prompt_tokens_id 中 prompt 所占据的字符个数，即从哪个偏移量之后开始， prompt_tokens_id 中的值才是我们真正的 request 信息。

完成了这一部分初始化后，使用 SequenceGroup 把 sequence 语句和我们的 prefix 一并包起来，并把 seq_group 放入我们的等待队列之中。
#### Sequence 对于 Blocks 的使用
由于 blocks 的使用相对重要，我们更加详细地查看这一过程：
```python
def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
    cursor = 0
    while cursor < len(token_ids):
        # 如果该 sequence 尚没有使用 blocks ，则通过 _append_logical_block 为它分配一个
        if not self.logical_token_blocks:
            self._append_logical_block()
        # 否则，检查最后一个 block 是否用完了
        last_block = self.logical_token_blocks[-1]
        if last_block.is_full():
            self._append_logical_block()
            last_block = self.logical_token_blocks[-1]
        # 再在这个 logic_block 中进行填充，这边的逻辑其实还是很清晰的
        num_empty_slots = last_block.get_num_empty_slots()
        last_block.append_tokens(token_ids[cursor:cursor +
                                            num_empty_slots])
        cursor += num_empty_slots

# 此处的 block_number 是通过一种递增的方式进行排布使用的
def _append_logical_block(self) -> None:
    block = LogicalTokenBlock(
        block_number=len(self.logical_token_blocks),
        block_size=self.block_size,
    )
    self.logical_token_blocks.append(block)
```
看的出来在 sequence 以及 sequence group 进行初始化的时候，我们使用的 logic_block_number 并不是真正指向某个特定的 block ，而是代表了某个 sequence 处理 prompt_tokens 时所采用的一个 blocks 排布序列，为了容下这么多个 tokens ，我们才尝试采用了多达 logic_block_number 这么多个 blocks。
#### Prefix 的设置
本质上是希望以 block_size 为粒度来对信息进行解析
```python
        # Check whether the input specifies prefix
        prefix = self.scheduler.prefix_pool.add_or_get_prefix(
            prompt_token_ids[:prefix_pos], lora_request.lora_int_id
            if lora_request else 0) if prefix_pos is not None else None

class PrefixPool:
    """Manages all the prompt prefixes.

    NOTE: This feature is experimental and may be replaced with automatic
        prefix caching in the future.

    Args:
        block_size: The block size of the executed model.

    Attributes:
        prefixes: A list of all the prefixes.
        block_size: The block size of the executed model.
    """

    def __init__(
        self,
        block_size: int,
    ) -> None:
        # TODO(zhuohan): Add a capacity limit to the prefix pool.
        self.prefixes: Dict[int, Prefix] = {}
        self.block_size = block_size
    # 使用 prefix 的时候，粒度应该为 block_size 的倍数，这样才能以 block 为单位处理我们的 tokens
    def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
        new_length = len(token_ids) // self.block_size * self.block_size
        return tuple(token_ids[:new_length])
    # 开了一个叫做 prefixes 的数组以 hash 的形式存放我们的以 block 为粒度的 prefix 信息
    def add_or_get_prefix(self, token_ids: Sequence[int],
                          lora_int_id: int) -> Optional[Prefix]:
        token_ids = self._truncate_token_ids(token_ids)
        if len(token_ids) == 0:
            # Prefix is empty.
            return None
        prefix = Prefix(token_ids, self.block_size)
        # 没有 lora_int_id 则设置参数为 0
        prefix_hash = hash((prefix, lora_int_id))
        if prefix_hash not in self.prefixes:
            self.prefixes[prefix_hash] = prefix
        return self.prefixes[prefix_hash]
```
### _run_engine
```python
def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
    # Initialize tqdm.
    # 显示处理的进度条
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, desc="Processed prompts")
    # Run the engine.
    outputs: List[RequestOutput] = []
    # llm_engine 所对应的三个队列还没处理完
    while self.llm_engine.has_unfinished_requests():
        # 单次运行的结果
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            # 如果某个 output 已经完成，则加入到 outputs 队列之中
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    pbar.update(1)
    if use_tqdm:
        pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    # 根据 request_id 对 outputs 做一个排序
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs
```
run_engine 本身没有特别出彩，简言之就是多次 step 运行试图找到可以在之后输出的那些请求，直到三条队列之中的全部请求都被执行完了。每次 step 可能能够进入 finished 状态的 output 输出不止一条。

这需要更加谨慎观察 step 的逻辑。
#### llm_engine.step
step 的工作在于调用 _run_workers 这一函数，并从中攫取出相应的输出信息，再把输出信息传递给 _process_model_outputs 函数。
```python
def step(self) -> List[RequestOutput]:
    """Performs one decoding iteration and returns newly generated results.

    .. figure:: https://i.imgur.com/sv2HssD.png
        :alt: Overview of the step function
        :align: center

        Overview of the step function.

    Details:
        - Step 1: Schedules the sequences to be executed in the next
            iteration and the token blocks to be swapped in/out/copy.

            - Depending on the scheduling policy,
                sequences may be `preempted/reordered`.
            - A Sequence Group (SG) refer to a group of sequences
                that are generated from the same prompt.

        - Step 2: Calls the workers to execute the model.
        - Step 3: Processes the model output. This mainly includes:

            - Decodes the relevant outputs.
            - Updates the scheduled sequence groups with model outputs
                based on its `sampling parameters` (`use_beam_search` or not).
            - Frees the finished sequence groups.

        - Finally, it creates and returns the newly generated results.

    Example:
        >>> # Please see the example/ folder for more detailed examples.
        >>>
        >>> # initialize engine and request arguments
        >>> engine = LLMEngine.from_engine_args(engine_args)
        >>> example_inputs = [(0, "What is LLM?",
        >>>    SamplingParams(temperature=0.0))]
        >>>
        >>> # Start the engine with an event loop
        >>> while True:
        >>>     if example_inputs:
        >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
        >>>         engine.add_request(str(req_id), prompt, sampling_params)
        >>>
        >>>     # continue the request processing
        >>>     request_outputs = engine.step()
        >>>     for request_output in request_outputs:
        >>>         if request_output.finished:
        >>>             # return or show the request output
        >>>
        >>>     if not (engine.has_unfinished_requests() or example_inputs):
        >>>         break
    """
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

    if not scheduler_outputs.is_empty():
        # Execute the model.
        all_outputs = self._run_workers(
            "execute_model",
            driver_kwargs={
                "seq_group_metadata_list": seq_group_metadata_list,
                "blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
                "blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
                "blocks_to_copy": scheduler_outputs.blocks_to_copy,
            },
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG)

        # Only the driver worker returns the sampling results.
        output = all_outputs[0]
    else:
        output = []

    return self._process_model_outputs(output, scheduler_outputs)
```
##### scheduler.schedule
相关代码如下所示：通过 _schedule 函数得到 scheduler_outputs ，但这个 scheduler_outputs 信息我们还不清楚，需要继续向下阅读代码来进行调研。
```python
def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
    # Schedule sequence groups.
    # This function call changes the internal states of the scheduler
    # such as self.running, self.swapped, and self.waiting.
    scheduler_outputs = self._schedule()

    # Create input data structures.
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for seq_group in scheduler_outputs.scheduled_seq_groups:
        seq_data: Dict[int, SequenceData] = {}
        block_tables: Dict[int, List[int]] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq_id = seq.seq_id
            seq_data[seq_id] = seq.data
            block_tables[seq_id] = self.block_manager.get_block_table(seq)

        seq_group_metadata = SequenceGroupMetadata(
            request_id=seq_group.request_id,
            is_prompt=scheduler_outputs.prompt_run,
            seq_data=seq_data,
            sampling_params=seq_group.sampling_params,
            block_tables=block_tables,
            lora_request=seq_group.lora_request,
            prefix=seq_group.prefix,
        )
        seq_group_metadata_list.append(seq_group_metadata)
    return seq_group_metadata_list, scheduler_outputs

# 核心函数 _schedule 如下所示
def _schedule(self) -> SchedulerOutputs:
    # Blocks that need to be swaped or copied before model execution.
    blocks_to_swap_in: Dict[int, int] = {}
    blocks_to_swap_out: Dict[int, int] = {}
    blocks_to_copy: Dict[int, List[int]] = {}

    # Fix the current time.
    now = time.monotonic()

    # Join waiting sequences if possible.
    # 如果没有处于 swapped 队列的元素
    if not self.swapped:
        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        # 统计一下当前处于 running 状态的 seqs 一共有多少个，即 num_curr_seqs 
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None
        seq_lens: List[int] = []

        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        leftover_waiting_sequences = deque()
        # 当 waiting 队列中还有 seqgroups 存在时
        while self.waiting:
            seq_group = self.waiting[0]
            # 找到 seq_group 之中处于 WAITING 状态的 seqs 们
            waiting_seqs = seq_group.get_seqs(
                status=SequenceStatus.WAITING)
            # SequenceGroup 是这么初始化的，至少一开始初始化的时候只有一个 Sequence 与之对应，并且状态为 Waiting
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_prompt_tokens = waiting_seqs[0].get_len()
            # 准备处理该 Sequence 所对应的 prompt_tokens_ids
            if num_prompt_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            # 判断 block_manager 中的空间是否满足 seq_group 分配资源的需求
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds the capacity of block_manager")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue
            # 这玩意儿默认情况是不开启的，无视掉他
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if lora_int_id > 0 and lora_int_id not in curr_loras and len(
                        curr_loras) >= self.lora_config.max_loras:
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    self.waiting.popleft()
                    continue

            # If the number of batched tokens exceeds the limit, stop.
            # 把对应的长度 num_prompt_tokens 放入到 seq_lens 数组中，最后 batched 处理的 tokens 总数算的是一个最大值
            new_seq_lens = seq_lens + [num_prompt_tokens]
            num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
            if (num_batched_tokens >
                    self.scheduler_config.max_num_batched_tokens):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break
            # 有多少 tokens 位置其实是需要 padding 的
            num_paddings = num_batched_tokens - sum(new_seq_lens)
            if num_paddings > self.scheduler_config.max_paddings:
                break
            seq_lens = new_seq_lens

            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            # 成功把 waiting 列，即我们上面解析的 seq_group 真正从 waiting 队列中 pop 释放出来
            self.waiting.popleft()
            # 为该 seq_group 分配足够的空间
            self._allocate(seq_group)
            self.running.append(seq_group)
            # 给当前正在跑的 num_curr_seqs 继续再加上一个队列，并告知 scheduled 我们的 seq_group 已经被调度过了（现在处于 RUNNING 状态）
            num_curr_seqs += num_new_seqs
            scheduled.append(seq_group)
        # lora 不开启时，该项目不存在
        self.waiting.extendleft(leftover_waiting_sequences)
        # 一开始都为 waiting 的时候，第一次进入时会直接让 scheduled 为非负，从而在这边返回了值
        # 最好的情况是对 scheduleroutputs做一个较为详细地打印工作
        if scheduled or ignored_seq_groups:
            scheduler_outputs = SchedulerOutputs(
                scheduled_seq_groups=scheduled,
                prompt_run=True,
                num_batched_tokens=len(seq_lens) *
                max(seq_lens) if seq_lens else 0,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=ignored_seq_groups,
            )
            return scheduler_outputs

    # NOTE(woosuk): Preemption happens only when there is no available slot
    # to keep all the sequence groups in the RUNNING state.
    # In this case, the policy is responsible for deciding which sequence
    # groups to preempt.
    self.running = self.policy.sort_by_priority(now, self.running)

    # Reserve new token slots for the running sequence groups.
    running: Deque[SequenceGroup] = deque()
    preempted: List[SequenceGroup] = []
    while self.running:
        seq_group = self.running.popleft()
        while not self.block_manager.can_append_slot(seq_group):
            if self.running:
                # Preempt the lowest-priority sequence groups.
                victim_seq_group = self.running.pop()
                self._preempt(victim_seq_group, blocks_to_swap_out)
                preempted.append(victim_seq_group)
            else:
                # No other sequence groups can be preempted.
                # Preempt the current sequence group.
                self._preempt(seq_group, blocks_to_swap_out)
                preempted.append(seq_group)
                break
        else:
            # Append new slots to the sequence group.
            self._append_slot(seq_group, blocks_to_copy)
            running.append(seq_group)
    self.running = running

    # Swap in the sequence groups in the SWAPPED state if possible.
    self.swapped = self.policy.sort_by_priority(now, self.swapped)
    if not preempted:
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        leftover_swapped = deque()

        while self.swapped:
            seq_group = self.swapped[0]
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if lora_int_id > 0 and lora_int_id not in curr_loras and len(
                        curr_loras) >= self.lora_config.max_loras:
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    self.swapped.popleft()
                    continue

            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self.swapped.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            num_curr_seqs += num_new_seqs
            self.running.append(seq_group)

        self.swapped.extendleft(leftover_swapped)

    # Each sequence in the generation phase only takes one token slot.
    # Therefore, the number of batched tokens is equal to the number of
    # sequences in the RUNNING state.
    num_batched_tokens = sum(
        seq_group.num_seqs(status=SequenceStatus.RUNNING)
        for seq_group in self.running)

    scheduler_outputs = SchedulerOutputs(
        scheduled_seq_groups=self.running,
        prompt_run=False,
        num_batched_tokens=num_batched_tokens,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        ignored_seq_groups=[],
    )
    return scheduler_outputs
```