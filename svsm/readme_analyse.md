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
### vc_init: VC 中断处理初始化
`VC`问题本质上是`VMM communication exception`问题。这是由`CPU`触发的中断，一般是在需要`hypervisor emulation`环节时才会触发的中断。

可以查看`AMD-ES`和`AMD SEV-SNP`中的一部分流程图，从而得到一个基本的认识。

至于`vc`的初始化，我们可以看到其基本框架如下：
```rust
pub fn vc_init() {
    let ghcb_pa: PhysAddr = pgtable_va_to_pa(get_early_ghcb());

    vc_establish_protocol();
    vc_register_ghcb(ghcb_pa);
}
```
函数`pgtable_va_to_pa`示意图如下所示：
```rust
/// Obtain physical address (PA) of a page given its VA
pub fn pgtable_va_to_pa(va: VirtAddr) -> PhysAddr {
    PhysAddr::new_truncate(va.as_u64() - SVSM_GVA_OFFSET.as_u64())
}
```
似乎在`SVSM`中一开始的地址转换较为轻松，只需要把虚拟地址的高`12`位全部清除掉就好了。当然这是页表等东东尚未启动的时候的情况。

首先我们得到`ghcb_pa`的物理地址。接下来利用函数`vc_establish_protocol`来对`GHCB`模块按照协议进行处理。
```rust
fn vc_establish_protocol() {
    let mut response: u64;

    // Request SEV information
    // GHCB_MSR_SEV_INFO_REQ是GHCB协议中专门用于REQ的位
    // 如果其值为2，则说明是这个功能，具体可以查看GHCB的参数表格。
    // 见2.3.1中的协议表格
    response = vc_msr_protocol(GHCB_MSR_SEV_INFO_REQ);

    // Validate the GHCB protocol version
    // 返回值须得是0x001，低十二位，检查合法与否。
    if GHCB_MSR_INFO!(response) != GHCB_MSR_SEV_INFO_RES {
        vc_terminate_ghcb_general();
    }

    // 检查response之中是否超出了大值和小值，支持的版本号是一串
    // 看样子是一串连续的数字，因此可以做这样的比较。
    if GHCB_MSR_PROTOCOL_MIN!(response) > GHCB_PROTOCOL_MAX
        || GHCB_MSR_PROTOCOL_MAX!(response) < GHCB_PROTOCOL_MIN
    {
        vc_terminate_ghcb_unsupported_protocol();
    }

    // Request hypervisor feature support
    // 0x80对应的功能，得到hypervisor feature support bitmap
    // 存放在GHCB INFO低十二位部分。
    // 高位被设置为0。
    response = vc_msr_protocol(GHCB_MSR_HV_FEATURE_REQ);

    // Validate required SVSM feature(s)
    // 理应得到的低12位返回值是0x81号数据，与上面相应成趣。作为返回值。
    if GHCB_MSR_INFO!(response) != GHCB_MSR_HV_FEATURE_RES {
        vc_terminate_ghcb_general();
    }

    // 对于返回的data值，检查其中的flag，必须要求其中的某一部分存在：
    // bit 0，bit 1两个部分是一定要存在的，其他无所谓
    // 不太清楚bit 4为什么也算在flag之中，我在GHCB中没有看到这个指示
    if (GHCB_MSR_HV_FEATURES!(response) & GHCB_SVSM_FEATURES) != GHCB_SVSM_FEATURES {
        vc_terminate_ghcb_feature();
    }

    // 将所支持的HV_FEATURES特性存储在这个名为HV的变量之中
    unsafe {
        HV_FEATURES = GHCB_MSR_HV_FEATURES!(response);
    }
}
```
函数`vc_msr_protocol`如下所示：
```rust
fn vc_msr_protocol(request: u64) -> u64 {
    let response: u64;

    // Save the current GHCB MSR value
    let value: u64 = rdmsr(MSR_GHCB);

    // Perform the MSR protocol
    wrmsr(MSR_GHCB, request);
    vc_vmgexit();
    response = rdmsr(MSR_GHCB);

    // Restore the GHCB MSR value
    wrmsr(MSR_GHCB, value);

    response
}
```
其中参数`request`的上文被设置为了`2`，恰好表示的意思是`REQUEST`相关。在这边利用指令rdmsr来讲`MSR_GHCB`固定需要投入的参数写入`rcx`寄存器，再把返回的结果写回到`value`之中，其中高位是`rdx`信息，低位是`rax`信息。

这一步的目的是把我们原本的值存放在`value`之中，然后我们就能把`request`信息写入到对应的寄存器中，以执行我们后续的一些操作。

但函数`vc_vmgexit`相对来说比较简单，其中似乎主要就是`rep vmmcall`一下就好了。
```rust
fn vc_vmgexit() {
    unsafe {
        asm!("rep vmmcall");
    }
}
```
最终返回`response`作为结果。我们利用得到该信息后，需要检查这个`GHCB protocol`返回的信息是否是合法的。由于整个`flag`的大小是`12`位，需要与`0xfff`进行`&`操作。

根据手册上的指示，返回值一定是`0x001`才能说明奏效了。这便是第一步的检验。之后，针对第一步检验后得到的返回值`GHCBData`信息需要进行第二部检验。

根据手册上的信息，我们可以得知：
```
0x001 – SEV Information
▪ GHCBData[63:48] specifies the maximum GHCB protocol version supported.
▪ GHCBData[47:32] specifies the minimum GHCB protocol version supported.
▪ GHCBData[31:24] specifies the SEV page table encryption bit number.
```
用户需要通过`GHCBData`中提供的版本号来确定`hypervisor`到底支持哪些版本，从而选择合适版本并对`GHCB Protocol`协议的版本做一些限定。如果`guest`没法支持`hypervisor`提供的协议范围，那就寄了，需要发送`0x100`表示终结我们的任务。

在完成了`vc_establish_protocol`函数之后，我们进入函数`vc_register_ghcb`之中。该函数的大体情况如下所示：
```rust
pub fn vc_register_ghcb(pa: PhysAddr) {
    // Perform GHCB registration
    let response: u64 = vc_msr_protocol(GHCB_MSR_REGISTER_GHCB!(pa.as_u64()));

    // Validate the response
    // 按照手册，这一步很自然。
    if GHCB_MSR_INFO!(response) != GHCB_MSR_REGISTER_GHCB_RES {
        vc_terminate_svsm_general();
    }

    // 这一步也相当自然。
    if GHCB_MSR_DATA!(response) != pa.as_u64() {
        vc_terminate_svsm_general();
    }

    // 不知道为什么，反正先把pa写到了rax和rcx之中。
    // 可能之后有用到。
    wrmsr(MSR_GHCB, pa.as_u64());
}
```
我们一点点来看，先分析`vc_msr_protocol`的参数，其根据`ghcb_pa`，即`early_ghcb`的物理地址来确定它的`request`，其中`GHCB_MSR_REGISTER_GHCB`宏的信息如下所示：
```rust
// MSR protocol: GHCB registration
/// 0x12
const GHCB_MSR_REGISTER_GHCB_REQ: u64 = 0x12;
macro_rules! GHCB_MSR_REGISTER_GHCB {
    ($x: expr) => {
        (($x) | GHCB_MSR_REGISTER_GHCB_REQ)
    };
}
```
在这边本质上是把`Request`与`pa`一起封装成`64`位的协议数据包，可以参考手册，其上恰是这么要求的。
```
 0x012 – Register GHCB GPA Request
▪ GHCBData[63:12] – GHCB GFN to register
Written by the guest to request the GHCB guest physical address (GHCB GPA 
= GHCB GFN << 12) be registered for the vCPU invoking the VMGEXIT. See
section 2.3.2 for further details and restrictions.
```
在2.3.2环节中，这一段代码对应的是后面的`Register`环节。把这个信息写入到`request`变量中后，根据手册，我们需要看到`0x13`作为`GHCBInfo`返回值。而`data`段`hypervisor`则需要用`GHCB GPA`来进行响应。

到这里，这个函数也分析完了。

### mem_init: 内存初始化
这个函数大体上如下所示：
```rust
/// Initialized the runtime memory allocator
///
/// The mem_init() function sets up the root memory region data structures so
/// that memory can be allocated and released. It will set up the page
/// meta-data information and the free-lists for every supported allocation order
/// of the buddy allocator.
/// It will also setup the SLAB allocator for allocations up to 2 KiB.
pub fn mem_init() {
    STATIC_ASSERT!(MAX_ORDER < REAL_MAX_ORDER);

    unsafe {
        __mem_init();
    }
}

unsafe fn __mem_init() {
    // 这边我们暂且不管这个动态地址是怎么做的，就把它当做实际上跑的时候的动态地址吧
    let pstart: PhysAddr = pgtable_va_to_pa(get_dyn_mem_begin());
    let pend: PhysAddr = pgtable_va_to_pa(get_dyn_mem_end());

    // 利用pstart和pend值来确定对应的PhysFrame物理页号
    let mem_begin: PhysFrame = PhysFrame::containing_address(pstart);
    let mem_end: PhysFrame = PhysFrame::containing_address(pend);

    // 将从mem_begin开始到mem_end结束的物理页号对应的物理内存设置为private状态
    vc_early_make_pages_private(mem_begin, mem_end);

    let vstart: VirtAddr = get_dyn_mem_begin();
    let page_count: usize = ((pend.as_u64() - pstart.as_u64()) / PAGE_SIZE) as usize;

    root_mem_init(pstart, vstart, page_count);
}
```
在这边我们需要研究一下函数`vc_early_make_pages_private`，如下所示：
```rust
// 该函数把Ghcb记作一个可变项
pub fn vc_early_make_pages_private(begin: PhysFrame, end: PhysFrame) {
    let ghcb: *mut Ghcb = get_early_ghcb().as_mut_ptr() as *mut Ghcb;

    perform_page_state_change(ghcb, begin, end, PSC_PRIVATE);
}

fn perform_page_state_change(ghcb: *mut Ghcb, begin: PhysFrame, end: PhysFrame, page_op: u64) {
    // 不太清楚这个数据结构具体做什么。
    /*
        #[repr(C, packed)]
        struct PscOp {
            pub header: PscOpHeader,
            pub entries: [PscOpData; PSC_ENTRIES],
        }
    */
    let mut op: PscOp = PscOp::new();

    // 利用begin和end来确定物理页的起始地址
    let mut pa: PhysAddr = begin.start_address();
    let pa_end: PhysAddr = end.start_address();

    // 逐页遍历
    // 其中的PsC数据结构大概是一个用于虚拟机页面状态更改的数据结构和操作类型。
    while pa < pa_end {
        op.header.cur_entry = 0;
        // 可以把op当做一个管理者
        // 在build_psc_entries内部实现了逐个页面遍历并且赋上page_op属性的操作
        // 在这边建立了op，以方便管理，提前为页面分配了相应的空间，并且以op的形式进行存储
        pa = build_psc_entries(&mut op, pa, pa_end, page_op);

        let last_entry: u16 = op.header.end_entry;

        // 如果传递的参数page_op是PSC_SHARED类型，则采用下面的方式继续拿给你pvalidate
        // 创建的时候首先还是设置为不合法状态，所以记作RESCIND
        if page_op == PSC_SHARED {
            pvalidate_psc_entries(&mut op, RESCIND);
        }

        let size: usize =
            size_of::<PscOpHeader>() + size_of::<PscOpData>() * (last_entry as usize + 1);
        unsafe {
            let set_bytes: *const u8 = &op as *const PscOp as *const u8;
            let get_bytes: *mut u8 = &mut op as *mut PscOp as *mut u8;

            // 对ghcb页面做清空处理，清空exit code以及valid_bitmap之中的内容
            (*ghcb).clear();

            // 对ghcb中的大小为size的空间设置为可以被共享的情况
            // size对应的部分其实是管理者PscOp的全部信息内容，说白了就是一个像管理向量一样的东西，把它设置成shared状态
            // 将op内用于管理的那些数据整体赋值到ghcb之中，数据长度为size，源数据为op，拷贝目标地址为ghcb
            (*ghcb).set_shared_buffer(set_bytes, size);

            while op.header.cur_entry <= last_entry {
                // 设置GHCB_NAE_PSC并退出，调用vmmcall指令
                // psc是svsm之上的组件
                vc_perform_vmgexit(ghcb, GHCB_NAE_PSC, 0, 0);
                if !(*ghcb).is_sw_exit_info_2_valid() || (*ghcb).sw_exit_info_2() != 0 {
                    vc_terminate_svsm_psc();
                }
                // 继续执行拷贝操作，东西记录在ghcb之中
                (*ghcb).shared_buffer(get_bytes, size);
            }
        }

        // 如一开始就打算作为私有，则很快就能把它设置为VALIDATE状态
        // 记作shared的GHCB本身应该被设置为RESINE状态
        if page_op == PSC_PRIVATE {
            op.header.cur_entry = 0;
            op.header.end_entry = last_entry;
            pvalidate_psc_entries(&mut op, VALIDATE);
        }
    }
}

// 可以把op当做一个管理者
// 在build_psc_entries内部实现了逐个页面遍历并且赋上page_op属性的操作
// 直到遍历完成，此时在op.entries[i]内将会存放各个页对应的信息，2MB和4KB不同的页面将会以不同的方式来进行存储
// 他们的区别可以通过GHCB_2MB等相关宏进行区分，这边做的方式很武断，直接 | 上一个2 ^ 56，从而彻底避开了干扰
fn build_psc_entries(op: &mut PscOp, begin: PhysAddr, end: PhysAddr, page_op: u64) -> PhysAddr {
    let mut pa: PhysAddr = begin;
    let mut i: usize = 0;

    while pa < end && i < PSC_ENTRIES {
        if pa.is_aligned(PAGE_2MB_SIZE) && (end - pa) >= PAGE_2MB_SIZE {
            op.entries[i].data = GHCB_2MB_PSC_ENTRY!(pa.as_u64(), page_op);
            pa += PAGE_2MB_SIZE;
        } else {
            op.entries[i].data = GHCB_4KB_PSC_ENTRY!(pa.as_u64(), page_op);
            pa += PAGE_SIZE;
        }
        op.header.end_entry = i as u16;

        i += 1;
    }

    return pa;
}

// pvalidate_psc_entries函数信息
// 对于psc_entries，利用pvalidate_op，通过op，进行pvalidate状态调整操作
fn pvalidate_psc_entries(op: &mut PscOp, pvalidate_op: u32) {
    let first_entry: usize = op.header.cur_entry as usize;
    let last_entry: usize = op.header.end_entry as usize + 1;

    // 从op中的header入手，逐项进行解析
    // 拆分掉header entry内我们先前内嵌的一些信息，从而精准获取gpa，size等信息
    // 最终目的是对于VM entry进行pvalidate化，毕竟pvalidate的对象应该是虚拟内存页面
    for i in first_entry..last_entry {
        let gpa: u64 = GHCB_PSC_GPA!(op.entries[i].data);
        let size: u32 = GHCB_PSC_SIZE!(op.entries[i].data);

        let mut va: VirtAddr = pgtable_pa_to_va(PhysAddr::new(gpa));
        // 对vm entry实现pvalidate操作，这是一串特殊的指令，采用.byte 0xf2, 0x0f, 0x01, 0xff的方式来进行表示
        // 其中最后返回值可以对应的状态大体如下：
        /*
            /// 1
            pub const PVALIDATE_FAIL_INPUT: u32 = 1;
            /// 6
            pub const PVALIDATE_FAIL_SIZE_MISMATCH: u32 = 6;

            根据指令集手册，一般只有0,1,6三种类型的返回值，因此下面的三种类型可能是其他扩展功能
            我们所能确定的一件事情倒是：最大提供的值似乎是0xf
            /// 15
            pub const PVALIDATE_RET_MAX: u32 = 15;
            /// 16
            pub const PVALIDATE_CF_SET: u32 = 16;
            /// 17
            pub const PVALIDATE_RET_ERR: u32 = 17;
        */
        let mut ret: u32 = pvalidate(va.as_u64(), size, pvalidate_op);
        if ret == PVALIDATE_FAIL_SIZE_MISMATCH && size > 0 {
            let va_end = va + PAGE_2MB_SIZE;

            while va < va_end {
                ret = pvalidate(va.as_u64(), 0, pvalidate_op);
                if ret != 0 {
                    break;
                }

                va += PAGE_SIZE;
            }
        }
        // 正常完成的返回值是0，其他的都不被允许，会采用各种各样的方式将它处理掉。
        if ret != 0 {
            vc_terminate_svsm_psc();
        }
    }
}
```
将`early_ghcb`所对应的虚拟地址设置完备后，我们现在看一下函数`root_mem_init`：
```rust
fn root_mem_init(pstart: PhysAddr, vstart: VirtAddr, page_count: usize) {
    {
        let mut region = ROOT_MEM.lock();
        region.start_phys = pstart;
        region.start_virt = vstart;
        region.page_count = page_count;
        // init_memory函数开辟了大小为PageStorageType的meta_pages，以及其余选项
        // 这里头的细节比较多，涉及SVSM页面设置上的一些细节，我们在分析mem_init函数时还是不要在这个地方做过多停留了。
        region.init_memory();
        // drop lock here so SLAB initialization does not deadlock
    }

    // 启用SLAB内存分配器
    if let Err(_e) = SLAB_PAGE_SLAB.lock().init() {
        panic!("Failed to initialize SLAB_PAGE_SLAB");
    }
}
```
`ROOT_MEM`是一块标准的地址区域，其包含信息如下：
```rust
/// Data structure representing a region of allocatable memory
///
/// The memory region must be physically and virtually contiguous and
/// implements a buddy algorithm for page allocations.
///
/// All allocations have a power-of-two size and are naturally aligned
/// (virtually). For allocations to be naturally aligned physically the virtual
/// and physical start addresses must be aligned at MAX_ORDER allocation size.
///
/// The buddy allocator takes some memory for itself to store per-page page
/// meta-data, which is currently 8 bytes per PAGE_SIZE page.
struct MemoryRegion {
    /// Physical start address
    start_phys: PhysAddr,
    /// Virtual start address
    start_virt: VirtAddr,
    /// Total number of PAGE_SIZE
    page_count: usize,
    /// Total number of pages in the region per ORDER
    nr_pages: [usize; MAX_ORDER],
    /// Next free page per ORDER
    next_page: [usize; MAX_ORDER],
    /// Number of free pages per ORDER
    free_pages: [usize; MAX_ORDER],
}

// 初始化过程则如下所示：看得出来也确实比较好玩
pub const fn new() -> Self {
    MemoryRegion {
        start_phys: PhysAddr::new(0),
        start_virt: VirtAddr::new_truncate(0),
        page_count: 0,
        nr_pages: [0; MAX_ORDER],
        next_page: [0; MAX_ORDER],
        free_pages: [0; MAX_ORDER],
    }
}
```
总而言之，在`mem_init`函数中，我们通过读`ghcb`块，并利用`pvalidate`命令将其设置为用户所私有的块，并对`root_mem`做了一些初始化工作。

这个代码分析流程似乎比我想象的漫长很多，我们还是悠着点吧。

还是不以这种粘贴的形式进行代码解读了，直接在原来的工程代码中做注释比较好。