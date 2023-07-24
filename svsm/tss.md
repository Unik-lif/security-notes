## 有关于函数tss_init的分析
我们在这一章尝试分析`percpu_init`函数。
```rust
/// Create and load TSS.
/// Only used by the BSP, since APs can use tss_init_for()
/// 仅仅被bsp来拿来使用
pub fn tss_init() {
    unsafe {
        __tss_init();
    }
}

unsafe fn __tss_init() {
    // 在内核上分配了tss空间，并且对IST表做了简单的初始化工作
    let tss: VirtAddr = create_tss();
    let tss_base: u64 = tss.as_u64();
    let tss_limit: u64 = (size_of::<TaskStateSegment>() - 1) as u64;

    // 从start.S处获取gdt_tss对应的地址位置
    // 应该是物理地址
    let gdt_tss0: *mut u64 = get_early_tss().as_u64() as *mut u64;
    let gdt_tss1: *mut u64 = (get_early_tss().as_u64() + 8) as *mut u64;
    // 这一部分的TSS信息似乎不怎么需要细看
    // 因为JOS我没看过，这部分肯定有点小糊涂
    // 我似乎只要知道，TSS的地址要存放在GS之中就可以了
    // Update existing TSS entry in the GDT.
    // 该选择子是X86中的零碎细节，除非在写JOS要搞懂，其他情况完全可以当做黑盒

    *gdt_tss0 = (SVSM_TSS_TYPE as u64) << 40;
    *gdt_tss0 |= (tss_base & 0xff000000) << 32;
    *gdt_tss0 |= (tss_base & 0x00ffffff) << 16;
    *gdt_tss0 |= tss_limit;

    *gdt_tss1 = tss_base >> 32;

    PERCPU.set_tss(tss);

    load_tss(SegmentSelector(get_gdt64_tss() as u16));
}

///
/// Create new TSS for a given CPU, but don't load it.
/// Used by AP creation where the VMSA can be used to pre-set the
/// task register (TR) with the TSS values
///
pub fn tss_init_for(cpu_id: usize) -> VirtAddr {
    let tss: VirtAddr;

    unsafe {
        tss = create_tss();
        PERCPU.set_tss_for(tss, cpu_id);
    }

    tss
}
```
核心函数为`__tss_init`，这个函数首先利用`create_tss`来创建一个空间。
```rust
unsafe fn create_tss() -> VirtAddr {
    // 分配大小为TaskStateSegment的一个空间，名为tss_va
    let tss_va: VirtAddr = match mem_allocate(size_of::<TaskStateSegment>()) {
        Ok(f) => f,
        Err(()) => vc_terminate_svsm_enomem(),
    };

    let tss: *mut TaskStateSegment = tss_va.as_mut_ptr();
    let tss_template: TaskStateSegment = TaskStateSegment::new();

    // Make sure we have correct initial values
    *tss = tss_template;
    // 这个部分的内存是TSS管理内存，由内核进行管理
    // mem_create_stack将会同时创建guard page和stack pages
    // 意思是会额外多分配一个Guard Page
    // 返回一个虚拟地址
    let ist_stack: VirtAddr = mem_create_stack(IST_STACK_PAGES, false);
    // 设置第一个interrupt stack pointer为刚刚创建的ist_stack
    // 其他的我们暂时不管
    (*tss).interrupt_stack_table[DOUBLE_FAULT_IST] = ist_stack;

    tss_va
}
```
`TaskStateSegment`的数据结构如下所示：其中存放了`Task`相关的敏感数据，包括中断与权限级栈表。
```rust
/// In 64-bit mode the TSS holds information that is not
/// directly related to the task-switch mechanism,
/// but is used for finding kernel level stack
/// if interrupts arrive while in kernel mode.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed(4))]
pub struct TaskStateSegment {
    reserved_1: u32,
    /// The full 64-bit canonical forms of the stack pointers (RSP) for privilege levels 0-2.
    pub privilege_stack_table: [VirtAddr; 3],
    reserved_2: u64,
    /// The full 64-bit canonical forms of the interrupt stack table (IST) pointers.
    pub interrupt_stack_table: [VirtAddr; 7],
    reserved_3: u64,
    reserved_4: u16,
    /// The 16-bit offset to the I/O permission bit map from the 64-bit TSS base.
    pub iomap_base: u16,
}
```

到这边，我们的代码分析结束了。简而言之`TSS`的初始化过程还是比较简单的。我们为`BSP`提供一个用于存放进程任务相关信息的`TSS`结构体，然后对它进行初始化操作，把相关的信息写入当前`CPU`所对应的`GS`寄存器中就好了。