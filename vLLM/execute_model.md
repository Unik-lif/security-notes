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
    # 从cpu_cache出发，swap_in放到gpu_cache之中
    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)
    # 从gpu_cache出发，swap_out放到cpu_cache之中
    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        # cache_stream由torch.cuda.Stream()生成
        # torch.cuda.Stream => wrapper around a CUDA stream.
        with torch.cuda.stream(self.cache_stream):
            # 在多重layers之间，对于每个layer都执行一次swap_blocks的操作
            for i in range(self.num_layers):
                # 每个层的数据存在不同的KVCache之中
                # KVCache是由两个torch.tensor类型的数据包裹而成
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(
                    src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(
                    src_value_cache, dst_value_cache, src_to_dst)
                event = self.events[i]
                # 对于每个层layer，都有一个event
                # 尝试记录之
                event.record(stream=self.cache_stream)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        # 我们需要mark一下copy_blocks这个函数，其涉及key_caches与value_caches的部分
        # 这边似乎是对于该worker下CacheEngine中的整个gpu-Cache进行操作
        # 而上面的swapped由于涉及更加细粒度的blocks层面的交互，调用的是一层layer的KVCache
        # 这边则是整个gpu-Cache，对应整个layers层的KVCache
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
```
在这一步的解析在注释中就差不多完毕了，可以看的出来真实情况下的`Cache`被定义为了`torch.Tensor`类型的数据，并且每个`layer`对应一个`KVCache`似乎有这个先验知识就够了。在这里我们对`KVCache`布局的总结如下：

根据`GPU_Device`，即板卡数目来确定当前一共有多少个`workers`，每个`worker`对应一个`CacheEngine`，每个`CacheEngine`对应`cpu-Cache`和`gpu-Cache`各一个，而一个`cpu-Cache`或者`gpu-Cache`都对应`layers`个`KVCache`（每个层级上都有一个`KVCache`），每个`KVCache`对应一个`Key-Cache`和一个`Value-Cache`，每个`Key-Cache`和`Value-Cache`都对应多个`Blocks`。

之后我们需要看一下`cuda`代码，检查`cache_ops`相关操作究竟做了什么。这个文件是`cache_kernels.cu`：

我们首先来看`swap_blocks`函数。该函数通过`setup.py`被导入到`pytorch`中，在`cu`这边，维持着与`python`那边一致的接口。换言之，`Cache`的调度机制是更低一层的抽象。
```Cpp
void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  // 确定src_device和dst_device所对应的具体设备
  // 参考pytorch文档，device()返回的类型一般是三种：cpu，cuda，mps
  // 在我们这边感觉更多是前两个
  // 对于gpu_cache的torch blocks，初始化时带上了`cuda`关键字
  // 对于cpu_cache的torch blocks，初始化时应该是当前处理浮点数的机器的类型，应该默认对应的是`cpu`关键字
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    // 跨GPU传递数据确实不太可行也不太可信
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  void *src_ptr = src.data_ptr();
  void *dst_ptr = dst.data_ptr();

  // src[0].numel() => how many columns.
  // element_size() => size of element, 32-bit type => 4.
  // take the first block as src[0] in a tensor(Cache)
  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  // for mapping blocks => blocks need to be swapped.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    // select the specific block.
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    // cuda层面的异步块复制，在cpu/gpu之间进行交互
    cudaMemcpyAsync(
      dst_ptr + dst_offset,
      src_ptr + src_offset,
      block_size_in_bytes,
      memcpy_type,
      stream);
  }
}
```
`swap_block`还是比较好理解的，通过`schedule`提供的`swap mappings`信息，在这边实打实地通过`pytorch`实现了`cpu`和`gpu`之间的块层面的`swap`。

接下来我们来调研`copy_blocks`这个函数：不同与上面立足于`block`的`swapping`（`block`是`tensor`的一个`row`，`tensor`对应一个层的`KVCache`，不过上面的函数调用的接口是一个`tensor`，也就是一个层级的`KVCache`），而`copy_blocks`这边则是立足于整个`gpu_cache`，对应于`layers`个层的`KVCache`，对应一个`tensor`的`vector`，也就是一个矩阵的集合。
```cpp
void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  // 每个layer分别对应每一层KVCache所对应的key_cache的起始地址
  // 在这边的循环结束后可以把这件事当做，我们目前把整个gpu_cache的key_cache和value_cache做了一个压扁化
  // 把原本由一个tensor vector所构成的东西压扁，目标是转化成一个很长很长的tensor，因此以让GPU所能够接收到它
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }
  // Create block mapping array.
  // 根据blocks_to_copy来进行处理
  std::vector<int> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    // 被替换的block, src_block_number
    int src_block_number = pair.first;
    // 替换src_block_number的由于copy-on-write机制的存在，大概率是一个
    // 相对比较长的块的序列，对于序列而言，需要通过遍历把对应的dst_block_number
    // 逐一压入到block_mapping_vec之中
    for (int dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  // 返回第一个元素的指针，也就是以array的形式把数据调取出来
  int* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  // 把先前导出来的数据结构以torch tensor形式组织，以让GPU也能够理解
  // 第一项是数据开始的指针，第二项是数组的长度，第三项是数据的element_size()
  // 依此实现torch的构造

  // 该worker所对应的key_cache的全部layers层合并后的数据信息
  torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  // 该worker所对应的value_cache的全部layers层合并后的数据信息
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  // 该worker所对应的确定进行copy_mapping，这个copy_mapping是为了copy-on-write机制服务，
  // 这是这里所存在的映射所构成的tensor
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt).to(cache_device);

  // Launch the kernel.
  // key_caches[0]: layer 0 key_cache
  // key_caches[0][0]: layer 0 key_cache block 0.
  // key_caches[0][0].numel(): layer 0 key_cache block 0 column numbers
  // 在这边对部分数据结构做了一个准备，之后准备启动cuda_kernel
  // 这一部分的代码比较难以理解，借用chatgpt力量

  /*
  const int numel_per_block = key_caches[0][0].numel();：这行代码计算了每个块（block）中元素的数量，通过访问 key_caches 数组中的第一个元素的第一个子元素，并调用 numel() 方法获取元素数量。

  dim3 grid(num_layers, num_pairs);：这行代码定义了 GPU 上的线程块（block）和网格（grid）的维度。grid 的维度为 (num_layers, num_pairs)，意味着将在 num_layers 行和 num_pairs 列的二维网格中执行计算。

  dim3 block(std::min(1024, numel_per_block));：这行代码定义了线程块（block）的维度。block 的维度为 (std::min(1024, numel_per_block))，即最多包含 1024 个线程。此处使用 std::min() 函数是为了确保线程块的大小不超过 1024。

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();：这行代码获取当前的 CUDA 流（stream），用于在 GPU 上执行异步操作。

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] { ... }))：这是一个宏，用于根据输入的数据类型选择不同的代码分发路径。它将根据 key_caches[0] 的标量类型选择适当的分发路径。分发路径的选择是通过调用名为 "copy_blocks_kernel" 的函数来实现的。

  vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(...)：这行代码启动了 GPU 上的核函数（kernel function），用于执行数据拷贝操作。vllm::copy_blocks_kernel 是要执行的核函数的名称，scalar_t 是模板参数，用于指定数据类型。<<<grid, block, 0, stream>>> 设置了核函数的执行配置，其中 grid 是网格的维度，block 是线程块的维度，stream 是要在其中执行核函数的 CUDA 流。

  key_cache_ptrs_tensor.data_ptr<int64_t>()、value_cache_ptrs_tensor.data_ptr<int64_t>() 和 block_mapping_tensor.data_ptr<int>()：这些是从之前代码中创建的 PyTorch 张量中获取数据指针的方法。这些指针将作为核函数的参数传递给 vllm::copy_blocks_kernel。 
  */

  const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    // 通过调用copy_blocks_kernel函数做分发处理
    key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
      vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key_cache_ptrs_tensor.data_ptr<int64_t>(),
        value_cache_ptrs_tensor.data_ptr<int64_t>(),
        block_mapping_tensor.data_ptr<int>(),
        numel_per_block);
    }));
}
```
现在我们尝试来分析做分发处理的函数`copy_blocks_kernel`，这个函数输入了`key_cache`、`value_cache`等信息，其根据`blockIdx`来确定当前的层数`index`和`mapping`中的`pair index`，但这边的`blockIdx`是什么，目前暂不知。总体来看，`blockIdx`对应的是`grid`，反映了网格大小，并且`grid`存放了`layer_idx`总量和`pair_idx`总量信息。

考虑到`CUDA`本身的并行化性质，可以合理推测下面的函数是把原本的工作分解成了仅针对于`layer_idx`这一层的拷贝工作？
```cpp
namespace vllm {

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int src_block_number = block_mapping[2 * pair_idx];
  int dst_block_number = block_mapping[2 * pair_idx + 1];

  const int src_block_offset = src_block_number * numel_per_block;
  const int dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

} // namespace vllm
```