## 这是什么东西？
首先声明一点，该函数并没有出现在`main`分支之上，出现在`cpl`分支，我们姑且来看看这个分支相对于`main`分支多做了什么事情。
### 对于syscall_init的分析
```rust
/// Prepare system calls and CPL switches
/// SYSCALL/SYSRET is already enabled in MSR EFER (see start/svsm.h)
pub fn syscall_init() {
    // GDT entries for SYSCALL/SYSRET
    wrmsr(MSR_STAR, syscall_gdt());

    // Disable interrupts when entering a system call
    wrmsr(MSR_SFMASK, SFMASK_INTERRUPTS_DISABLED);

    // Register system call entry point
    wrmsr(MSR_LSTAR, get_syscall_entry().as_u64());
}

/// Returns what MSR STAR should have to allow syscall/sysret
pub fn syscall_gdt() -> u64 {
    // Needed for SYSRET
    let gdt_user32_cs: u64 = get_gdt64_user32_cs() << 48;

    // Needed for SYSCALL
    let gdt_kernel_cs: u64 = get_gdt64_kernel_cs() << 32;

    gdt_user32_cs | gdt_kernel_cs
}
```
看得出来其简单地对`gdt`等相关物件做了初始化。此外，似乎没什么好深究的。
### cpl_init函数
```rust
/// Prepare user code for CPL switching
pub fn cpl_init() {
    lazy_static::initialize(&USER_INFO_MUTEX);
}
```
这只是初始化了一个可以给用户使用的锁而已。
### cpl_go_unprivileged函数
这个函数似乎实现了从`CPL0`向`CPL3`的跳转工作。
```rust
/// Jump from CPL0 to CPL3
/// Information on userspace will be printed if on verbose mode
pub fn cpl_go_unprivileged() {
    let user_stack_va: VirtAddr;
    let user_code_va: VirtAddr;
    let user_code_va_end: VirtAddr;

    // Create stack for userspace
    // Because mem_create_stack() adds guard page, move end address
    // 给用户一个包含Guard Page的USER_STACK以供使用
    user_stack_va = mem_create_stack(USER_STACK_SIZE, true) - PAGE_SIZE;

    (user_code_va, user_code_va_end) = user_code_va_addr();

    jump_to_user(user_stack_va, user_code_va, user_code_va_end);
}
```
之后，利用函数`user_code_va_addr`来锁定虚拟地址中的`data`段范围：
```rust
/// Retrieve addresses for user code start and end
fn user_code_va_addr() -> (VirtAddr, VirtAddr) {
    let user_code_va: VirtAddr = get_svsm_edata();
    let user_code_end_va: VirtAddr = get_svsm_size();
    (user_code_va, user_code_end_va)
}
```
再尝试通过函数`jump_to_user`跳转到用户态之中：
```rust
fn jump_to_user(user_stack_va: VirtAddr, user_code_va: VirtAddr, user_code_end: VirtAddr) {
    // Make stack point to its end
    let stack_size: u64 = PAGE_SIZE * USER_STACK_SIZE;
    let code_pages: u64 = ((user_code_end.as_u64() - user_code_va.as_u64()) / PAGE_SIZE) + 1;
    // 根据user_code_end来计算用户需要的代码段大小
    // 不过这似乎是一个新准备开的空间，本来只是从user code的起始和结束部分计算了一个小区间
    let new_ucode_va: VirtAddr = VirtAddr::new(get_cpl3_start());
    let new_ucode_end: VirtAddr =
        VirtAddr::new(get_cpl3_start() + (code_pages - 1_u64) * PAGE_SIZE);
    // 现在计算得到了给cpl3分配的start位置开始的一部分内存空间
    // 当然现在似乎只是虚拟地址上的计算，并没有真实进行映射和分配
    let new_stack_va: VirtAddr = new_ucode_va - stack_size - PAGE_SIZE;
    // 映射这一部分空间，把edata开始的一些代码段信息装载过来
    // 也同时从物理内存中分配相应的页面，进行拷贝
    // 这一部分之所以是代码段，是因为内部包含了名为executable的flag
    if map_user_code(new_ucode_va, pgtable_va_to_pa(user_code_va), code_pages) == false {
        prints!("Required page table updates for user code failed!\n");
        vc_terminate_svsm_page_err();
    }
    // 分配栈空间，这边的空间被设置无法executable
    if map_user_stack(
        new_stack_va,
        pgtable_va_to_pa(user_stack_va),
        USER_STACK_SIZE,
    ) == false
    {
        prints!("Required page table updates for user stack failed!\n");
        vc_terminate_svsm_page_err();
    }

    // Print userspace information of this CPU
    // 打印用户信息，主要对应的是一些栈和用户代码段的信息
    print_user_info(
        new_stack_va.as_u64(),
        new_stack_va.as_u64() + stack_size,
        new_ucode_va.as_u64(),
        new_ucode_end.as_u64(),
        stack_size,
        code_pages,
    );
    // 在这边设置一些关于中断表之类的东西，并从特权状态退出，进入到用户态
    iretq(
        get_gdt64_user64_ds(),
        new_stack_va.as_u64(),
        get_gdt64_user64_cs(),
        new_ucode_va.as_u64(),
    );
}
```
在这个函数的末尾通过`iretq`从特权级跳转到了用户态，而之前内核已经给用户提供了一定大小的栈空间和代码空间以运行。

这边完成了从内核态向用户态的跳转，那么从用户态的视角来看这件事情，应该是怎么样的呢？
### cpl3相关
`cpl3`的启动源码相较于`cpl0`少了很多东西，其不再需要通过一些`BSP`相关的检查。
```rust
pub fn user_request_loop() {
    loop {
        // Ask kernel to start listening for guest requests and
        // get back to us when there is a request that userspace
        // can handle

        let (protocol, _callid) = get_next_request();
        let rax: u64;

        match protocol {
            _ => rax = SVSM_ERR_UNSUPPORTED_PROTOCOL,
        }

        // Update vmsa.rax with the result and mark call as completed
        set_request_finished(rax);
    }
}
```
通过函数`get_next_request`来获取系统调用，并加以运行。之后便进行无限调用。
```rust
#[inline]
fn get_next_request() -> (u32, u32) {
    let request: u64 = system_call!(SystemCalls::GetNextRequest as u32);
    // 这边的request排布方式非常符合SVSMprotocol的基本排布方式
    (UPPER_32BITS!(request) as u32, LOWER_32BITS!(request) as u32)
}
```
至此，关于`cpl`的代码解读完成，但是整个系统我们尚未完全将必要的组件搭建起来。我们所疑惑的问题如下所示：

1. `cpl0`与`cpl3`之间通过`syscall`来进行交互，这个`syscall`是怎样被`SVSM`所拦截并且拿去解析成对应的核心协议的。
2. 其他的组件是怎么搭上去的
3. 或许有必要来看看`coco-svsm`