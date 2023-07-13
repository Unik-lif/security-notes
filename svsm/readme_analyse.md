## SVSM的设计理念
Linux SVSM提供了一个交流接口，该接口能够把敏感指令交给拥有特权的用户来处理。

Linux SVSM基于SNP和SEV技术：这边在之后建议查阅相关的文档。

### 如何做到权限的分离？
在AMD SNP中引入了VMPL特权级用于强化安全层面的控制。VMPL0拥有最高的权利，Linux SVSM运行在VMPL 0上，而其他用户则只能运行在VMPL >= 1的层面上。这样自体系结构上来看，SVSM天生使得其余的用户没了这个能耐。
### 所需支持：
SEV-SNP, Qemu, OVMF BIOS, Compatible guest.

## 启动：
一些细节在本人提供的文件
尝试解读start.S文件：

在这个文件中我们首先看到了下面的几个定义：
```C
#include "svsm.h"

#define CBIT(x)			(BIT(51) + x)

#define SVSM_PGD_ENTRY(x)	(CBIT(x) + 0x03)
#define SVSM_P4D_ENTRY(x)	(CBIT(x) + 0x03)
#define SVSM_PUD_ENTRY(x)	(CBIT(x) + 0x03)
#define SVSM_PMD_ENTRY(x)	(CBIT(x) + 0x83)
#define SVSM_PTE_ENTRY(x)	(CBIT(x) + 0x03)
```
这边似乎对页表的位置做了一些定义，我们看到这是一个五级页表结构。当然，它的结构其实是自上而下（最高层是PGD，以此类推）。每个部分占9个bit，最后给4KB的页表留下12bit，这一部分和高位多的7bit用作FLAG存储地区。具体可以查看手册：

intel 开发者手册：2934/4830

https://lwn.net/Articles/717293/

在上面的连接和intel手册中我们可以知道：
```
Systems running with five-level paging will support 52-bit physical addresses and 57-bit virtual addresses.
```
在这篇post发出时，到现在其实也就五年多，可以得知这确实是一个新的技术。

在SVSM启动时，首先运行脚本start.S中的相关内容，完成最开始的GHCB shared过程。在完成了一些零碎的初始化过程之后，我们的程序会进入svsm_main函数之中。

## main函数简析
```rust
/// Main function. Initialize everything and start request loop.
/// This function never returns.
#[no_mangle]
pub extern "C" fn svsm_main() -> ! {
    // Ensure valid SVSM execution environment
    initial_checks();

    // Initialize exception/interrupt handling
    idt_init();

    // Prepare VC handler
    vc_init();

    mem_init();

    // Create 4-level page table and load it
    pgtable_init();

    // Allocate per-CPU data (pointed to by GS register)
    percpu_init();

    // Set up the TSS
    tss_init();

    ghcb_init();

    serial_init();

    fwcfg_init();

    // Initialize and start APs
    smp_init();

    // Load BIOS
    start_bios();

    // Start taking requests from guest in this vCPU
    svsm_request_loop();

    // We should never reach this point
    loop {
        halt()
    }
}
```
我们逐个步骤进行分析：

### 初始检查initl_checks：
在这一部分中，对`SVSM`和相关的权限等级进行检查。
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
    // rmp_4k表示正常大小页，也有2M这种大页的用法。
    // vc_terminate是紫砂指令，会终止我们的任务。
    let ret: u32 = rmpadjust(get_svsm_begin().as_u64(), RMP_4K, VMPL::Vmpl1 as u64);
    if ret != 0 {
        vc_terminate(SVSM_REASON_CODE_SET, SVSM_TERM_NOT_VMPL0);
    }
}
```
这一步的检查抓住了一个核心，即仅仅是`VMPL0`的权限级才能进行`RMPADJUST`指令的赋权操作。

对于函数`check_vmpl0_features`，其对应实现如下所示：
```rust
/// Check SVSM is running with adequate SEV features
fn check_vmpl0_features() {
    let features: u64 = rdmsr(MSR_SEV_STATUS) >> 2;

    if features & VMPL0_REQUIRED_SEV_FEATS != VMPL0_REQUIRED_SEV_FEATS {
        vc_terminate_vmpl0_sev_features();
    }

    if features & VMPL0_UNSUPPORTED_SEV_FEATS != 0 {
        vc_terminate_vmpl0_sev_features();
    }
}
```
太细节的地方我们还是不考虑了，在这边这个函数将会检查当前运作时利用`rdmsr`读取出出来的`features`值是符合要求的，即对应的应该开启的SEV特性全被开启了。

下面的函数检查在`start.s`环节是否按照预期对我们的内存空间进行了相关的初始化。
```rust
/// Check addresses are appropriately aligned and within boundaries
fn check_svsm_address() {
    let total_size: u64 = get_svsm_end().as_u64() - get_svsm_begin().as_u64();
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
有一点奇怪，get_svsm_end这些函数我没找到在哪里出现过。

在这边考虑了两件事，一个是是否动态分配内存空间时依旧在我们一开始给定内存的范围之中。第二件事情便是对齐与否。

### 中断处理：idt_init
`x86`一般处理中断的流程，是把中断向量表直接加载进来给特定硬件来使用。
```rust
/// Load IDT with function handlers for each exception
pub fn idt_init() {
    IDT.load();
}
```
我们先来看看`IDT`是什么东西。
```rust
lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt: InterruptDescriptorTable = InterruptDescriptorTable::new();

        unsafe {
            idt.double_fault
                .set_handler_fn(df_handler)
                .set_stack_index(DOUBLE_FAULT_IST as u16);
        }
        idt.general_protection_fault.set_handler_fn(gp_handler);
        idt.page_fault.set_handler_fn(pf_handler);
        idt.vmm_communication_exception.set_handler_fn(vc_handler);

        idt
    };
}
```
我们可以看到这里用了很经典的`lazy_static`方法，并利用`x86-64`模块中的部分组件来实现，当然，他们使用的模块是经过自己修改后的版本，内部嵌入了关于`#VC`、`#GP`等错误处理器放置的地方。在`unsafe`模块中设置了`df_handler`作为`double_fault`的处理者。

其他异常处理采用的初始化方式类似。最后利用`lidt`指令装载进去。
### vc_init: VC 中断处理
`VC`问题本质上是`VMM communication exception`问题。这是由`CPU`触发的中断，一般是在需要`hypervisor emulation`环节时才会触发的中断。

可以查看`AMD-ES`和`AMD SEV-SNP`中的一部分流程图，从而得到一个基本的认识。

至于`vc`的初始化，我们可以看到其基本框架如下：
