## 有关于函数ghcb_init的分析
我们在这一章尝试分析`ghcb_init`函数。

```rust
pub fn ghcb_init() {
    // 首先，ghcb的大小理应和页表的大小保持一致。
    STATIC_ASSERT!(size_of::<Ghcb>() == PAGE_SIZE as usize);

    //
    // Perform GHCB allocation in a loop to avoid allocation order failures
    // for large vCPU counts.
    //
    // percpu_count会去找CPU_COUNT中的内容
    // 这个变量的值在__percpu_init函数中得到了更新
    let count: usize = percpu_count();
    for i in 0..count {
        // mem_allocate_frame分配一个frame大小的物理地址空间
        let frame: PhysFrame = match mem_allocate_frame() {
            Some(f) => f,
            None => vc_terminate_svsm_enomem(),
        };
        let va: VirtAddr = pgtable_pa_to_va(frame.start_address());

        // 将从va开始的大小为PAGE_SIZE的内存空间页设置为shared状态
        pgtable_make_pages_shared(va, PAGE_SIZE);
        memset(va.as_mut_ptr(), 0, PAGE_SIZE as usize);

        unsafe {
            // 绑定该ghcb与CPU i在一起
            // 说白了就是设置对应的i号cpu的PerCpu数据结构的ghcb为va.
            PERCPU.set_ghcb_for(va, i);

            if i == 0 {
                // Register the BSPs GHCB
                // 此处的ghcb不同于以往，之前的early_ghcb仅仅用于初始化
                // 现在这边的是要真正投入使用了，因此需要简单注册一下下
                vc_register_ghcb(frame.start_address());
            }
        }
    }
}
```
这个函数很简单，为所有的`cpu`都分配一个`ghcb`空间，并同时给BSP注册掉它的`ghcb`。