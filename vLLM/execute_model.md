## 概览：
在这篇博客中，我们尝试分析一下完成调度之后，把调度结果送给`workers`之后，`workers`的具体行为。
## 代码分析，从step出发
以前看过最大的项目是`rCore`和`linux-svsm`，虽然`vLLM`不至于特别复杂，但对于代码能力堪忧的我来说感觉还是足够喝一壶。

我们在`scheduler`一栏目中，系统梳理了`vLLM`中的`scheduler`机制，现在我们继续之前的工作。
```python
def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # 返回了metadata和scheduler_outputs
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        # 那确实没什么好说的，什么都不用做
        if (not seq_group_metadata_list) and scheduler_outputs.is_empty():
            # Nothing to do.
            return []

        # Execute the model.
        # _run_workers是一个调用workers的接口，可以当做函数调用的启动器
        # 这边叫起来了execute_model函数，之后我们会分析一下 ①
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # Update the scheduler with the model outputs.
        # 根据结构，更新scheduler中的情况
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs
```
### execute_model分析
其实这个函数似乎很简单，由于我们先前已经在`blocks_to_swap_in`等地方记录了块交换的重要信息，接下来只是通过这个接口来尝试把原本存在于`gpu_allocator`和`cpu_allocator`与真正去完成工作的`workers`中的`cache_engine`建立起一定的联系。
```python
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # Issue cache operations.
        # 触发cache相关的操作
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True
        # cache_events在初始化的时候就定下来了
        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        # 根据metadata_list准备一些数据
        # 这个感觉可能和AI有关系了
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        # 跑一遍model
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output
```
我们需要具体观察三个`cache_engine`操作函数，我们依此来做这件事情，首先`swap_in`与`swap_out`这两个行为是对称的，所以我们可以省下一些力气。
```python
    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        # 我们需要mark一下copy_blocks这个函数，其涉及key_caches与value_caches的部分
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
```
