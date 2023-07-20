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



### mem_init: 内存初始化
