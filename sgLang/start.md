在这一系列博文中，我们尝试简单分析一下`sgLang`的源码。

首先看向示例中的启动流程，可以看到下面这一步：
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```
因此入口在launch_server这边，尝试对此进行研究：

```
python -m sglang.launch_server --model-path TheBloke/Llama-2-70B-Chat-GPTQ --port 10005
```

看起来还不能直接通过大水漫灌的方式来构造驱逐集，得想办法`study`一下相关的代码和特性。

先前尝试通过阅读代码来提高构造驱逐集的效率，但是不行，得重新开始来阅读相关的代码。

我们分析的sglang时0.2.6版本的。
## 内存信息跟踪
### 初始化内存
首先在init_memory_pool位置建立：
```python
def init_memory_pool(self, total_gpu_memory, max_num_reqs=None):
    # max_total_num_tokens反映了显存大小总量
    self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)

    if self.max_total_num_tokens <= 0:
        raise RuntimeError(
            "Not enough memory. Please try to increase --mem-fraction-static."
        )

    if max_num_reqs is None:
        max_num_reqs = max(
            int(self.max_total_num_tokens / self.model_config.context_len * 512),
            2048,
        )

    # ReqToTokenPool和TokenToKVPool这两个部分
    # Map a request to its token locations
    self.req_to_token_pool = ReqToTokenPool(
        max_num_reqs,
        self.model_config.context_len + 8,
    )
    # Map a token to its kv cache locations
    self.token_to_kv_pool = TokenToKVPool(
        self.max_total_num_tokens,
        dtype=self.dtype,
        head_num=self.model_config.get_num_kv_heads(self.tp_size),
        head_dim=self.model_config.head_dim,
        layer_num=self.model_config.num_hidden_layers,
    )
    # 同时根据上面的max_total_num_tokens来确认了max_num_reqs，req_to_token_pool，token_to_kv_pool等变量的值
    logger.info(
        f"[gpu_id={self.gpu_id}] Memory pool end. "
        f"avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
    )

# profile_max_num_token
def profile_max_num_token(self, total_gpu_memory):
    available_gpu_memory = get_available_gpu_memory(
        self.gpu_id, distributed=self.tp_size > 1
    )
    head_dim = self.model_config.head_dim
    head_num = self.model_config.get_num_kv_heads(self.tp_size)
    cell_size = (
        head_num
        * head_dim
        * self.model_config.num_hidden_layers
        * 2
        * torch._utils._element_size(self.dtype)
    )
    # 剩下的内存大小 => max_num_token，在单卡的情况下，available_gpu_memory和total_gpu_memory的值是一样的
    # 因此这边本质上应该就是被分配为静态空间的显存大小
    rest_memory = available_gpu_memory - total_gpu_memory * (
        1 - self.mem_fraction_static
    )
    max_num_token = int(rest_memory * (1 << 30) // cell_size)
    return max_num_token
```
在model_runner这边做好初始化之后，需要在ModelTpServer这边将相关信息落实到位。
```python
class ModelTpServer:
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_overide_args: dict,
    ):
        suppress_other_loggers()

        # Copy arguments
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.schedule_heuristic = server_args.schedule_heuristic
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_overide_args=model_overide_args,
        )
        # 在这边连上刚刚初始化完毕的modelrunner信息
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

        if is_multimodal_model(server_args.model_path):
            self.processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
        # max_total_num_tokens从model_runner位置获取
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        # 这个值从max_prefill_tokens获取，不过它是先前设置好的
        # 这个值似乎和其他的GPU信息没有太大关联
        self.max_prefill_tokens = (
            16384
            if server_args.max_prefill_tokens is None
            else server_args.max_prefill_tokens
        )
        # max_running_requests的情况也是从max_total_num_tokens，即GPU显存总量有相关的
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size - 1,
        )
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size)
        )
        self.max_req_input_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        set_random_seed(server_args.random_seed)

        # Print info
        logger.info(
            f"[gpu_id={self.gpu_id}] "
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init cache
        # 内存在初始化的时候就已经有max_token_num的影子了
        self.tree_cache = RadixCache(
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            disable=server_args.disable_radix_cache,
        )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.scheduler = ScheduleHeuristic(
            self.schedule_heuristic,
            self.max_running_requests,
            self.max_prefill_tokens,
            self.max_total_num_tokens,
            self.tree_cache,
        )
        # 这两个值均是通过max_token_num来计算得到的
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Init running status
        self.forward_queue: List[Req] = []
        self.running_batch: Batch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()

        # Init the FSM cache for constrained generation
        self.regex_fsm_cache = FSMCache(
            server_args.tokenizer_path,
            {
                "tokenizer_mode": server_args.tokenizer_mode,
                "trust_remote_code": server_args.trust_remote_code,
            },
        )
        self.jump_forward_cache = JumpForwardCache()

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.min_new_token_ratio = min(
            global_config.base_min_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.new_token_ratio = self.min_new_token_ratio
        self.new_token_ratio_decay = global_config.new_token_ratio_decay
        self.new_token_ratio_recovery = global_config.new_token_ratio_recovery
```
到这边可以算是把初始化搞好了，现在关键回到什么时候操作了上面的数据结构。
### 目标
我们希望能够达到和flush_cache类似的效果，也就是下面这个样子：
```python
def flush_cache(self):
    if len(self.forward_queue) == 0 and (
        self.running_batch is None or len(self.running_batch.reqs) == 0
    ):
        self.tree_cache.reset()
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.regex_fsm_cache.reset()
        self.req_to_token_pool.clear()
        self.token_to_kv_pool.clear()
        torch.cuda.empty_cache()
        logger.info("Cache flushed successfully!")
    else:
        warnings.warn(
            f"Cache not flushed because there are pending requests. "
            f"#queue-req: {len(self.forward_queue)}, "
            f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
        )
```
需要破坏和影响到的部分包括萨皇宫内面的全部部分，应该是可以做到`torch.cuda.empty_cache`的？但是上面的部分需要通过发送大量的请求来实现。
### RadixCache
重点放在RadixCache中的一些接口是否被调用，以及被谁调用了。

一些观察：
- 本身这些接口，比如insert和match_prefix似乎都不需要观察具体的显存信息

#### cache_req
但是cache_req这个函数可能比较特殊：是使用RadixCache的一个核心函数。
```python
def cache_req(
    self,
    token_ids,
    last_uncached_pos,
    req_pool_idx,
    del_in_memory_pool=True,
    old_last_node=None,
):
    # Insert the request into radix cache
    # req_to_token: (size, max_context_len) 这样的张量
    # maps a request to its token location，记录相关的tokens信息位置到indices之中存放
    indices = self.req_to_token_pool.req_to_token[req_pool_idx, : len(token_ids)]
    new_prefix_len = self.insert(token_ids, indices.clone())

    if self.disable:
        if del_in_memory_pool:
            self.token_to_kv_pool.free(indices)
        else:
            return torch.tensor([], dtype=torch.int64), self.root_node

    # Radix Cache takes one ref in memory pool
    # 增大了can_use_mem_size的大小，并设置indices[last_uncached_pos:new_prefix_len]这个位置在mem_state的状态为True
    # 然后由于只用一个（保证没有重复的token内存），所以现在可能就删掉其中一个备份。
    self.token_to_kv_pool.free(indices[last_uncached_pos:new_prefix_len])

    # 默认走这条路，会自动清空这一部分，在handle_finished_requests中使用，在这个过程中，同时清空了req_to_token_pool的值
    if del_in_memory_pool:
        self.req_to_token_pool.free(req_pool_idx)
    else:
        # 先看有哪些部分是在radixcache之中存在的
        cached_indices, new_last_node = self.match_prefix(token_ids)
        assert len(cached_indices) == len(token_ids)
        # 在req_to_token_pool中做更新，以从req映射到token_pool
        self.req_to_token_pool.req_to_token[
            req_pool_idx, last_uncached_pos : len(cached_indices)
        ] = cached_indices[last_uncached_pos:]
        # 把老的请求进行一次dec操作，更新evictable_size
        self.dec_lock_ref(old_last_node)
        # 对新的请求进行以此inc操作，更新evictable_size
        self.inc_lock_ref(new_last_node)
        return cached_indices, new_last_node
```
#### evict相关
```python
def evict(self, num_tokens, evict_callback):
    if self.disable:
        return

    # 一路收集叶子节点
    leaves = self._collect_leaves()
    # 最小堆处理，根据last_access_time，按照LRU规律来排序
    heapq.heapify(leaves)

    num_evicted = 0
    # 需要驱逐num_tokens这么多的token单元
    while num_evicted < num_tokens and len(leaves):
        x = heapq.heappop(leaves)

        if x == self.root_node:
            break
        # 只驱逐目前没在被使用的
        if x.lock_ref > 0:
            continue

        # callback函数是之后可能会被使用的
        evict_callback(x.value)
        num_evicted += len(x.value)
        # 真正删掉子节点
        self._delete_leaf(x)

        # 如果孩子不够用，自己补上
        if len(x.parent.children) == 0:
            heapq.heappush(leaves, x.parent)
```
下面的是记录结点使用情况的函数：
```python
# 一路进行lock_ref的修改，溯流而上
def inc_lock_ref(self, node: TreeNode):
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.value)
            delta -= len(node.value)
        node.lock_ref += 1
        node = node.parent
    return delta

def dec_lock_ref(self, node: TreeNode):
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:
            self.evictable_size_ += len(node.value)
            delta += len(node.value)
        node.lock_ref -= 1
        node = node.parent
    return delta

def evictable_size(self):
    return self.evictable_size_
```
#### 调用情况
首先看到函数cache_req的情况：
```python
# 根据当前的Batch信息，对其中的的信息在里头通过cache_req来修改RadixCache情况
def cache_filled_batch(self, batch: Batch):
    req_pool_indices_cpu = batch.req_pool_indices.cpu().numpy()
    for i, req in enumerate(batch.reqs):
        new_prefix_indices, new_last_node = self.tree_cache.cache_req(
            token_ids=tuple(req.origin_input_ids + req.output_ids)[:-1],
            last_uncached_pos=len(req.prefix_indices),
            req_pool_idx=req_pool_indices_cpu[i],
            del_in_memory_pool=False,
            old_last_node=req.last_node,
        )
        req.prefix_indices, req.last_node = new_prefix_indices, new_last_node

# 在完成之后，删除memory_pool中req_to_token的映射
def handle_finished_requests(self, batch: Batch):
    ## --------------------
    # Remove finished reqs
    if finished_indices:
        # Update radix cache
        req_pool_indices_cpu = batch.req_pool_indices.tolist()
        for i in finished_indices:
            req = batch.reqs[i]
            self.tree_cache.cache_req(
                token_ids=tuple(req.origin_input_ids + req.output_ids)[:-1],
                last_uncached_pos=len(req.prefix_indices),
                req_pool_idx=req_pool_indices_cpu[i],
            )

            self.tree_cache.dec_lock_ref(req.last_node)

        # Update batch tensors
        if unfinished_indices:
            batch.filter_batch(unfinished_indices)
        else:
            batch.reqs = []
```

### 鸟瞰
从一开始的函数调用栈来进行分析：直接看向start_controller_process_single函数到loop_for_forward函数，从exposed_step
```python
def exposed_step(self, recv_reqs):
    try:
        # Recv requests
        # 仅仅支持两种方式来进行工作
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                self.handle_generate_request(recv_req)
            elif isinstance(recv_req, FlushCacheReq):
                self.flush_cache()
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req)
            else:
                raise ValueError(f"Invalid request: {recv_req}")

        # Forward
        self.forward_step()
    except Exception:
        logger.error("Exception in ModelTpServer:\n" + get_exception_traceback())
        raise

    # Return results
    ret = self.out_pyobjs
    self.out_pyobjs = []
    return ret


# handle_generate_request函数，并没有涉及关键的内存池相关操作
def handle_generate_request(
    self,
    recv_req: TokenizedGenerateReqInput,
):
    req = Req(recv_req.rid, recv_req.input_text, recv_req.input_ids)
    req.pixel_values = recv_req.pixel_values
    if req.pixel_values is not None:
        req.pad_value = [
            (recv_req.image_hash) % self.model_config.vocab_size,
            (recv_req.image_hash >> 16) % self.model_config.vocab_size,
            (recv_req.image_hash >> 32) % self.model_config.vocab_size,
            (recv_req.image_hash >> 64) % self.model_config.vocab_size,
        ]
        req.image_size = recv_req.image_size
        (
            req.origin_input_ids,
            req.image_offset,
        ) = self.model_runner.model.pad_input_ids(
            req.origin_input_ids_unpadded,
            req.pad_value,
            req.pixel_values.shape,
            req.image_size,
        )
    req.sampling_params = recv_req.sampling_params
    req.return_logprob = recv_req.return_logprob
    req.logprob_start_len = recv_req.logprob_start_len
    req.top_logprobs_num = recv_req.top_logprobs_num
    req.stream = recv_req.stream
    req.tokenizer = self.tokenizer

    # Init regex fsm
    # 这边通过一个regex_fsm_cache函数来进行一个基本的cache，用来记录命中的基本情况
    if req.sampling_params.regex is not None:
        req.regex_fsm = self.regex_fsm_cache.query(req.sampling_params.regex)
        if not self.disable_regex_jump_forward:
            req.jump_forward_map = self.jump_forward_cache.query(
                req.sampling_params.regex
            )

    # Truncate prompts that are too long
    if len(req.origin_input_ids) >= self.max_req_input_len:
        logger.warn(
            "Request length is longer than the KV cache pool size or "
            "the max context length. Truncated!!!"
        )
        req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]
    req.sampling_params.max_new_tokens = min(
        (
            req.sampling_params.max_new_tokens
            if req.sampling_params.max_new_tokens is not None
            else 1 << 30
        ),
        self.max_req_input_len - 1 - len(req.origin_input_ids),
    )
    self.forward_queue.append(req)
```
关键函数应该是forward_step：
```python
@torch.inference_mode()
def forward_step(self):
    # 
    new_batch = self.get_new_prefill_batch()

    if new_batch is not None:
        # Run a new prefill batch
        # 如果can_run_list不是空的，也就是说available_size大小还是足够的
        # 然后forward_prefill_batch会真的在memory pool中，根据前面get_new_prefill_batch的函数来分配相关的memory pool资源
        self.forward_prefill_batch(new_batch)
        # 重新cache_filled_batch把剩下的请求通过cache_req压入
        self.cache_filled_batch(new_batch)
        # 保证running_batch一直有东西
        if not new_batch.is_empty():
            if self.running_batch is None:
                self.running_batch = new_batch
            else:
                self.running_batch.merge(new_batch)
    else:
        # Run a decode batch
        if self.running_batch is not None:
            # Run a few decode batches continuously for reducing overhead
            for _ in range(global_config.num_continue_decode_steps):
                self.num_generated_tokens += len(self.running_batch.reqs)
                self.forward_decode_batch(self.running_batch)

                # Print stats
                if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
                    self.print_stats()

                if self.running_batch.is_empty():
                    self.running_batch = None
                    break

                if self.out_pyobjs and self.running_batch.has_stream():
                    break
        else:
            self.check_memory()
            self.new_token_ratio = global_config.init_new_token_ratio
```
下面我们对该关键函数的部分进行分析：
#### get_new_prefill_batch
```python
def get_new_prefill_batch(self) -> Optional[Batch]:
    # 正在跑的batch size
    running_bs = (
        len(self.running_batch.reqs) if self.running_batch is not None else 0
    )
    if running_bs >= self.max_running_requests:
        return

    # Compute matched prefix length
    for req in self.forward_queue:
        req.input_ids = req.origin_input_ids + req.output_ids
        # match_prefix其实已经改变了radix_cache的树的结构
        prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
        if req.return_logprob:
            prefix_indices = prefix_indices[: req.logprob_start_len]
        req.extend_input_len = len(req.input_ids) - len(prefix_indices)
        req.prefix_indices = prefix_indices
        req.last_node = last_node

    # Get priority queue
    # 通过相关调度策略来得到，接下来要新拿出去处理的东西
    self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)

    # Add requests if there is available space
    can_run_list = []
    new_batch_total_tokens = 0
    new_batch_input_tokens = 0

    available_size = (
        self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
    )
    # 对于running_batch内部的东西，之后available_size中要腾出来这一部分
    # 这是正在进行处理的部分，这部分内存没法使用
    if self.running_batch:
        available_size -= sum(
            [
                (r.sampling_params.max_new_tokens - len(r.output_ids))
                * self.new_token_ratio
                for r in self.running_batch.reqs
            ]
        )
    
    # 对应的是即将前提的那一部分请求
    # 可以考虑具体情况，来决定是否把它放到running_batch中
    for req in self.forward_queue:
        if req.return_logprob and req.normalized_prompt_logprob is None:
            # Need at least two tokens to compute normalized logprob
            if req.extend_input_len < 2:
                delta = 2 - req.extend_input_len
                req.extend_input_len += delta
                req.prefix_indices = req.prefix_indices[:-delta]
                if req.image_offset is not None:
                    req.image_offset += delta
        if req.extend_input_len == 0 and req.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            req.extend_input_len = 1
            req.prefix_indices = req.prefix_indices[:-1]
            if req.image_offset is not None:
                req.image_offset += 1

        if (
            req.extend_input_len
            + req.sampling_params.max_new_tokens
            + new_batch_total_tokens
            < available_size
            and (
                req.extend_input_len + new_batch_input_tokens
                <= self.max_prefill_tokens
                or len(can_run_list) == 0
            )
        ):
            delta = self.tree_cache.inc_lock_ref(req.last_node)
            available_size += delta

            if not (
                req.extend_input_len
                + req.sampling_params.max_new_tokens
                + new_batch_total_tokens
                < available_size
            ):
                # Undo locking
                # 如果extend后的空间，available_size不够用了，不能锁在这边，那没有办法，需要
                # 先把这个东西整个拖出去，尝试释放掉对应的空间
                # 但是这一步可能还没到真释放这个空间，只是在索引上提出，这样有可能之后调用eviction_memory时能把这一部分内存释放掉
                delta = self.tree_cache.dec_lock_ref(req.last_node)
                available_size += delta
                break
            else:
                # Add this request to the running batch
                # 如果空间足够的话，则把他们放到running batch之中
                can_run_list.append(req)
                new_batch_total_tokens += (
                    req.extend_input_len + req.sampling_params.max_new_tokens
                )
                new_batch_input_tokens += req.extend_input_len
        else:
            break

        # 正在跑的和我们觉得能跑的
        if running_bs + len(can_run_list) >= self.max_running_requests:
            break

    if len(can_run_list) == 0:
        return None

    # Print stats
    if self.tp_rank == 0:
        hit_tokens = sum(len(x.prefix_indices) for x in can_run_list)
        self.tree_cache_metrics["total"] += (
            hit_tokens + new_batch_input_tokens
        ) / 10**9
        self.tree_cache_metrics["hit"] += hit_tokens / 10**9
        tree_cache_hit_rate = (
            self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
        )
        logger.info(
            f"[gpu_id={self.gpu_id}] Prefill batch. "
            f"#new-seq: {len(can_run_list)}, "
            f"#new-token: {new_batch_input_tokens}, "
            f"#cached-token: {hit_tokens}, "
            f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
            f"#running-req: {running_bs}, "
            f"#queue-req: {len(self.forward_queue) - len(can_run_list)}"
        )

    # Return the new batch
    # 新的batch就是can_run_list对应的内容
    new_batch = Batch.init_new(
        can_run_list,
        self.req_to_token_pool,
        self.token_to_kv_pool,
        self.tree_cache,
    )
    self.forward_queue = [x for x in self.forward_queue if x not in can_run_list]
    return new_batch
```
#### forward_prefill_batch
在new_batch不为空的时候，将会有能力调用这个函数：
```python
def forward_prefill_batch(self, batch: Batch):
    # Build batch tensors
    batch.prepare_for_extend(
        self.model_config.vocab_size, self.int_token_logit_bias
    )

    # Forward and sample the next tokens
    if batch.extend_num_tokens != 0:
        # 前面已经把相关的内存空间腾出来了，现在的问题是，如何处理，所以这边用了forward和sample这样的操作
        output = self.model_runner.forward(batch, ForwardMode.EXTEND)
        next_token_ids = batch.sample(output.next_token_logits)

        # Move logprobs to cpu
        if output.next_token_logprobs is not None:
            output.next_token_logprobs = output.next_token_logprobs[
                torch.arange(len(next_token_ids), device=next_token_ids.device),
                next_token_ids,
            ].tolist()
            output.input_token_logprobs = output.input_token_logprobs.tolist()
            output.normalized_prompt_logprobs = (
                output.normalized_prompt_logprobs.tolist()
            )

        next_token_ids = next_token_ids.tolist()
    else:
        next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)

    # Check finish conditions
    pt = 0
    for i, req in enumerate(batch.reqs):
        req.completion_tokens_wo_jump_forward += 1
        req.output_ids.append(next_token_ids[i])
        req.check_finished()

        if req.return_logprob:
            self.add_logprob_return_values(i, req, pt, next_token_ids, output)
            pt += req.extend_input_len

    self.handle_finished_requests(batch)

# 主要函数
# prepare_for_extend:
def prepare_for_extend(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
    device = "cuda"
    bs = len(self.reqs)
    # 有多少req，就在req_to_token_pool中分配多少
    req_pool_indices = self.req_to_token_pool.alloc(bs)


    # Allocate memory
    seq_lens, prefix_lens = np.array(seq_lens), np.array(prefix_lens)
    extend_num_tokens = seq_lens.sum() - prefix_lens.sum()
    # 要多增加这么多的空间，以完成token_to_kv_pool的分配
    # out_cache_loc是分配后之后的内存location
    out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
    # 如果alloc_size分配失败了
    if out_cache_loc is None:
        # 从tree_cache所给出的结构，来索引到token_to_kv_pool中去
        # 希望能够释放掉extend_num_tokens大小的空间
        # 用token_to_kv_pool.free来做callback函数，顺带把这些空间都清理了
        # 如果没有这个函数，那只是修改了原来的记录，而没有在真正的pool中做处理
        # 如果先前发现似乎大小不够，就已经在Radix Cache的Tree上做了ref的减少，因此这边可以直接evict
        self.tree_cache.evict(extend_num_tokens, self.token_to_kv_pool.free)
        out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)

        if out_cache_loc is None:
            print("Prefill out of memory. This should never happen.")
            self.tree_cache.pretty_print()
            exit()

# In Class TokenToKVPool
class TokenToKVPool:
    def alloc(self, need_size: int):
        # 可能是通过预取来提高效率，但是具体功能暂不明白
        # 可以被预取的话，直接存在buffer里就好了
        buffer_len = len(self.prefetch_buffer)
        if need_size <= buffer_len:
            select_index = self.prefetch_buffer[:need_size]
            self.prefetch_buffer = self.prefetch_buffer[need_size:]
            return select_index

        # 否则一部分放在prefetcher中，剩下的部分需要被分配
        addition_size = need_size - buffer_len
        # 以chunk_size作为最小单位，每次更多分配，需要至少分配chunk_size这么个大小
        alloc_size = max(addition_size, self.prefetch_chunk_size)
        select_index = (
            torch.nonzero(self.mem_state).squeeze(1)[:alloc_size].to(torch.int32)
        )

        if select_index.shape[0] < addition_size:
            return None

        # 这一部分空间已经被分配了，除非之后通过free来给他释放开来
        self.mem_state[select_index] = False
        # 消耗了这一部分空间，是因为扩大了prefeth_buffer空间，否则没法存下来
        self.can_use_mem_size -= len(select_index)

        self.prefetch_buffer = torch.cat((self.prefetch_buffer, select_index))
        ret_index = self.prefetch_buffer[:need_size]
        # 从新的起始点开始
        self.prefetch_buffer = self.prefetch_buffer[need_size:]

        return ret_index


# Handle_finished_requests
def handle_finished_requests(self, batch: Batch):
    # Omit here.

    # Remove finished reqs
    if finished_indices:
        # Update radix cache
        req_pool_indices_cpu = batch.req_pool_indices.tolist()
        for i in finished_indices:
            req = batch.reqs[i]
            # 在这边通过del，来清除已经完成的那些请求所对应的tokens和reqs对应的memory pool
            # 将其释放开来
            self.tree_cache.cache_req(
                token_ids=tuple(req.origin_input_ids + req.output_ids)[:-1],
                last_uncached_pos=len(req.prefix_indices),
                req_pool_idx=req_pool_indices_cpu[i],
            )

            # 这边的话则是修改引用的结构
            self.tree_cache.dec_lock_ref(req.last_node)

        # Update batch tensors
        if unfinished_indices:
            batch.filter_batch(unfinished_indices)
        else:
            batch.reqs = []
```
从上面可以看出，在处理一个Batch级别的请求时，一旦完成了相关请求，就真的会通过cache_req中的del_in_memory_pool来对相关请求所占据的token indices和reqs本身的内存进行释放。

但如果真的只这么做，是否会出现没有办法缓存得当的问题？比如说就没有这个缓存效果了？finished_reqs是否真的在memory_pool里头没有用武之地了？
#### cache_filled_batch
```python
def cache_filled_batch(self, batch: Batch):
    req_pool_indices_cpu = batch.req_pool_indices.cpu().numpy()
    for i, req in enumerate(batch.reqs):
        new_prefix_indices, new_last_node = self.tree_cache.cache_req(
            token_ids=tuple(req.origin_input_ids + req.output_ids)[:-1],
            last_uncached_pos=len(req.prefix_indices),
            req_pool_idx=req_pool_indices_cpu[i],
            del_in_memory_pool=False,
            old_last_node=req.last_node,
        )
        req.prefix_indices, req.last_node = new_prefix_indices, new_last_node
```
这边好像把batch中还没有跑完的那些reqs拿去重新填写tree_cache。

prefill的过程：
- 分配内存空间，尝试找到满足当前内存空间的最大batch，用来准备做prefill的处理
- 拿去做完prefill的过程