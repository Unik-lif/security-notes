## 有关initial_checks的笔记整理：
本来打算一口气在一则推文中将`SVSM`分析清楚，想着还是不要一口气吃成胖子比较好，因此我们这边还是稍微少做一点点。

姑且分析一下`intial_checks`中涉及的检查。
### 在这一节值得参考的代码
- `rmpadjust`函数的使用方法

### initial_checks干了什么：
在这一部分中，对`SVSM`和相关的权限等级进行检查。

似乎首先检查了`VMPL`等级，`VMPL0`对应等级的`features`是否被满足，以及检查`svsm`的地址情况。
```rust
/// Perform initial checkings to ensure adequate execution.
/// This means checking SVSM runs on VMPL0, with proper addresses
/// and sizes, and proper SEV features activate
fn initial_checks() {
    // Ensure execution at VMPL0
    check_vmpl_level();

    // Ensure we are running with proper SEV features
    check_vmpl0_features();

    // Ensure SVSM addresses and sizes are appropiate
    check_svsm_address();
}
```
### 函数check_vmpl_level
对于函数`check_vmpl_level`来说，其实现如下所示：
```rust
/// Use the RMPADJUST instruction to determine if the SVSM is executing at VMPL0
fn check_vmpl_level() {
    // Use the RMPADJUST instruction to determine if the SVSM is executing
    // at VMPL0. The RMPADJUST instruction can only update the attributes
    // of a lower VMPL-level (e.g.: VMPL0 can change VMPL1, VMPL2 or VMPL3).
    // By attempting to change the VMPL1 attributes of a page, it can be
    // determined if the SVSM is executing at VMPL0.
    //
    // Attempt to clear the VMPL1 attributes of the early GHCB page.
    // 如同注释所述：仅有VMPL0才有权限能够去修改VMPL1对于某个页的操作权限
    // 可能值得探讨的地方：手册上似乎说VMPL1也可以去动VMPL2和VMPL3之类的，或许可以考虑写demo测试一下
    // 
    // vc_terminate是紫砂指令，会终止我们的任务。
    
    let ret: u32 = rmpadjust(get_svsm_begin().as_u64(), RMP_4K, VMPL::Vmpl1 as u64);
    if ret != 0 {
        vc_terminate(SVSM_REASON_CODE_SET, SVSM_TERM_NOT_VMPL0);
    }
}
```
这一步的检查抓住了一个核心，即仅仅是`VMPL0`的权限级才能进行`RMPADJUST`指令的赋权操作。

我们看一下`rmpadjust`指令，其对于`svsm_begin`一开始位置对应的虚拟地址页采用了`rmpadjust`指令进行权限修改，
```rust
/// Update RMP (Reverse Map Table) with new attributes
pub fn rmpadjust(va: u64, page_size: u32, attrs: u64) -> u32 {
    let ret: u32;

    unsafe {
        asm!(".byte 0xf3,0x0f,0x01,0xfe",
             in("rax") va, in("rcx") page_size, in("rdx") attrs,
             lateout("rax") ret,
             options(nostack));
    }

    ret
}
```
`rmpadjust`指令可以从`AMD`对应的系统编程手册中得到对应的寄存器信息。`va`对应着寄存器`rax`，`rdx`对应着属性，这个属性的设置相对复杂，主要分成四个部分，请参考资料。

总而言之，在这里我们熟悉了`rmpadjust`指令的使用流程。
### check_vmpl0_features函数
对于函数`check_vmpl0_features`，其对应实现如下所示：`rdmsr`指令本身也是在`vmpl0`下才可以运行的指令，`ecx`寄存器将会存放`MSR number`。`RDMSR`指令是否被支持需要检查一下`cpuid`的情况。
```rust
/// Check SVSM is running with adequate SEV features
fn check_vmpl0_features() {
    // 可以在手册中找到MSR_SEV_STATUS的具体返回值
    let features: u64 = rdmsr(MSR_SEV_STATUS) >> 2;

    // 为什么在VMPL0下某些性质没法满足？我在手册上没有找到，不过这些对应的bit位确实搞清楚是什么东西了
    // 这边是需要满足的性质
    if features & VMPL0_REQUIRED_SEV_FEATS != VMPL0_REQUIRED_SEV_FEATS {
        vc_terminate_vmpl0_sev_features();
    }

    // 这边是暂时还没有被满足的性质
    if features & VMPL0_UNSUPPORTED_SEV_FEATS != 0 {
        vc_terminate_vmpl0_sev_features();
    }
}
```
太细节的地方我们还是不考虑了，在这边这个函数将会检查当前运作时利用`rdmsr`读取出出来的`features`值是符合要求的，即对应的`vmpl0`权限级下应该开启的`SEV`特性全被开启了，以及不应该开启的哪些`SEV`性质全部都是关着的。大致如此。

如果不满足情况，则通过`terminate`方式利用`GHCB`中与`Hypervisor`的交互协议进行紫砂操作。

### check_svsm_address函数 
下面的函数检查在`start.s`环节是否按照预期对我们的内存空间进行了相关的初始化。
```rust
/// Check addresses are appropriately aligned and within boundaries
fn check_svsm_address() {
    // 这减来减去如果按照正常情况，total_size中最终返回的值是256 MB大小
    let total_size: u64 = get_svsm_end().as_u64() - get_svsm_begin().as_u64();
    // 检查起始地址和总内存大小之和是否被2 MB对齐了。这边的对齐算法应该没啥问题，不过我感觉有空可以考虑尝试证明一下。
    if !PAGE_2MB_ALIGNED!(get_svsm_begin().as_u64()) || !PAGE_2MB_ALIGNED!(total_size) {
        vc_terminate_svsm_general();
    }
    // svsm_end is SVSM_GVA + SVSM_MEM. dyn_mem_begin is calculated based on
    // edata, so make sure it is within boundaries
    if get_svsm_end() < get_dyn_mem_begin() {
        vc_terminate_svsm_general();
    }
}
```
有一点奇怪，`get_svsm_end`这些函数我没找到在哪里出现过。但是现在我找到了，他们都作为一个宏定义在`global.rs`之中得到了声明。

在这边考虑了两件事，一个是是否动态分配内存空间时依旧在我们一开始给定内存的范围之中。第二件事情便是对齐与否。

到这边，第一步的检查就完成了。

### 总结，第一步检查了一些基本的权限设置、权限对应的`features`是否得到满足、`svsm`的地址是否得到了对齐且合法这三种基本属性。