## 有关于函数percpu_init的分析
在完成了内存的初始化和页表的初始化之后，我们进入函数`percpu_init`。
```rust
/// Allocate a per-CPU data page, pointed to by GS regs. Also obtain Apic
/// Ids (needed for SMP).
pub fn percpu_init() {
    // Start with a enough pages for one piece of per-CPU data
    let init_count: u64 = PAGE_COUNT!(PERCPU_SIZE);
    let init_frame: PhysFrame = match mem_allocate_frames(init_count) {
        Some(f) => f,
        None => vc_terminate_svsm_enomem(),
    };
    let count: u64;

    // 将init_frame位置处的地址写入到GS寄存器之中
    // a per-CPU data page is now pointed to by GS regs.
    // 这边是起始位置
    wrmsr(
        MSR_GS_BASE,
        pgtable_pa_to_va(init_frame.start_address()).as_u64(),
    );

    // 通过va相关信息，完成apic_id的全部获取，得知到底有多少个vCPU可以利用
    // 在SVSM中CPU与vCPU一一对应
    // 在得知ap核总数后，为他们的各自的PERCPU信息分配内存空间，并记录他们的信息。
    unsafe {
        count = __percpu_init(init_frame, init_count);
    }

    // 大概是释放掉多余的空间
    // 这一部分空间是在初始化的时候用到的内存空间，在我们已经运行好函数__percpu_init的时候，它已经没有存在的必要性了。
    if count != init_count {
        mem_free_frames(init_frame, init_count);
    }
}
```
首先我们来看一下`init_count`，即以页的角度来看，每个CPU所要求的空间大小是多少，在这里需要了解`PERCPU_SIZE`大致是多少。
```rust
#[repr(C)]
#[derive(Debug)]
///
/// Each CPU runs one vCPU, which mainly needs to save execution context
/// (Vmsa) and Caa
///
pub struct PerCpu {
    cpu_id: u32,
    apic_id: u32,

    ghcb: u64,

    vmsa: [u64; VMPL::VmplMax as usize],
    caa: [u64; VMPL::VmplMax as usize],

    tss: u64,
}
```
为了减小上下文的切换成本，`SVSM`中的每个`CPU`都会运行一个`vCPU`与之对应，之后用`mem_allocate_frames`来进行分配这么大的内存空间，分配成功即可将对应的`frame`号返回到`init_frame`之中。

在现在我们完成了`GS`寄存器向`init_frame`位置处的`per-CPU data`的指向工作，接下来我们进入函数`__percpu_init`来看看后续的操作。
```rust
unsafe fn __percpu_init(init_frame: PhysFrame, init_count: u64) -> u64 {
    // Place BSP early GHCB into per-CPU data for use in VC
    // BSP是启动用的核心
    let va: VirtAddr = get_early_ghcb();
    // 将PERCPU中存放的ghcb地址位置先找到，然后把它设置成va
    // 即early_ghcb所对应的地址
    PERCPU.set_ghcb(va);

    // Retrieve the list of APIC IDs
    // cpuid: fn 0000_000b 且 rcx 为 0 时的子函数
    // 通过rdx得到当前BSP的APIC_ID号
    let bsp_apic_id: u32 = get_apic_id();

    // 通过bsp_apic_id号得到全部的apic_ids，反映ap对应的id号们
    let apic_ids: Vec<u32> = vc_get_apic_ids(bsp_apic_id);
    CPU_COUNT = apic_ids.len();

    // 一共要存储多少Page空间，用count作为变量进行存储
    let count: u64 = PAGE_COUNT!(apic_ids.len() as u64 * PERCPU_SIZE);
    // 这边的frame是count为基准的、完成了初始化之后的最终位置
    let frame: PhysFrame;
    if count != init_count {
        frame = match mem_allocate_frames(count) {
            Some(f) => f,
            None => vc_terminate_svsm_enomem(),
        };
    } else {
        frame = init_frame;
    }

    // 把用于管理percpu信息的frame地址存放在PERCPU_VA之中，并且通过
    // wrmsr的方式，将这个信息写入到GS寄存器内，表示更新
    PERCPU_VA = pgtable_pa_to_va(frame.start_address());
    wrmsr(MSR_GS_BASE, PERCPU_VA.as_u64());

    PERCPU.set_cpu_id(0);
    PERCPU.set_apic_id(bsp_apic_id);
    PERCPU.set_ghcb(va);
    // 可以把PERCPU当做一个数组的第一个元素，其对应的是bsp_apic_id即boot时启动的一个core
    let mut cpu: u32 = 1;
    for i in 0..CPU_COUNT {
        if apic_ids[i] == bsp_apic_id {
            continue;
        }
        // 这一部分反映的是除了boot以外的AP core等信息
        PERCPU.set_cpu_id_for(cpu, i);
        PERCPU.set_apic_id_for(apic_ids[i], i);
        cpu += 1;
    }

    count
}
```
对于`PERCPU`变量，其相关函数如下所示：我们可以轻松通过`offset_mem`找到对应的`ghcb`的地址与相关的偏移量。
```rust
impl PerCpu {
    pub const fn new() -> Self {
        PerCpu {
            cpu_id: 0,
            apic_id: 0,

            ghcb: 0,
            vmsa: [0; VMPL::VmplMax as usize],
            caa: [0; VMPL::VmplMax as usize],

            tss: 0,
        }
    }
}
```
函数在通过`CPUID`相关指令获得了`BSP`，即启动时所用的处理器的核心的之后，调用了函数`vc_get_apic_ids`，以获得其余`APs`们的`ID`号，对应的函数`vc_get_apic_ids`如下所示：
```rust
pub fn vc_get_apic_ids(bsp_apic_id: u32) -> Vec<u32> {
    let mut apic_ids: Vec<u32>;
    let ghcb: *mut Ghcb = vc_get_ghcb();
    let pages: u64;

    unsafe {
        (*ghcb).set_rax(0);
        // GHCB_NAE_GET_APIC_IDS是SVSM专用的Exit code
        // Guest OS请求Host以vCPU的身份去执行SVSM
        // 具体这个功能能干什么我没有在资料中找到
        // 0x8000_0017与0x8000_0018目前都没有在GHCB的Specification中提及
        // 需要对照源代码进行一些阅读工作，包括在info设置的值不同时对应的功能是否会发生变化
        // 在这边的例子中还是很明显功能发生变化了的
        // 需要阅读Hypervisor相关的源码
        vc_perform_vmgexit(ghcb, GHCB_NAE_GET_APIC_IDS, 0, 0);

        if !(*ghcb).is_rax_valid() {
            vc_terminate_svsm_resp_invalid();
        }
        // 一共有 rax 个核心
        // 每个核心要分配一个页用来使用
        pages = (*ghcb).rax();

        (*ghcb).clear();
    }
    
    let frame: PhysFrame = match mem_allocate_frames(pages) {
        Some(f) => f,
        None => vc_terminate_svsm_enomem(),
    };
    // va数目由pa来确定，pa数目由frames起始位置来决定
    let pa: PhysAddr = frame.start_address();
    let va: VirtAddr = pgtable_pa_to_va(pa);
    // 将va开始的这么一大块页设置为shared状态
    pgtable_make_pages_shared(va, pages * PAGE_SIZE);
    memset(va.as_mut_ptr(), 0, (pages * PAGE_SIZE) as usize);

    unsafe {
        (*ghcb).set_rax(pages);

        vc_perform_vmgexit(ghcb, GHCB_NAE_GET_APIC_IDS, pa.as_u64(), 0);

        if !(*ghcb).is_rax_valid() {
            vc_terminate_svsm_resp_invalid();
        }

        if (*ghcb).rax() != pages {
            vc_terminate_svsm_resp_invalid();
        }

        (*ghcb).clear();
        // count数目由va来确定
        // 既然已经在之前通过memset来做了，那么只有一种可能了，在GHCB_NAE_GET_APIC_IDS这个协议中，pa作为参数改变了va存储的值
        // 否则我们无法得知为何*count会发生改变并被设置成特殊的值
        let count: *const u32 = va.as_u64() as *const u32;

        if *count == 0 || *count > 4096 {
            vc_terminate_svsm_resp_invalid();
        }
        // 根据count来确定有多少个apic_ids
        apic_ids = Vec::with_capacity(*count as usize);

        // BSP is always CPU 0
        apic_ids.push(bsp_apic_id);
        for i in 0..*count {
            let id: *const u32 = (va.as_u64() + 4 + (i as u64 * 4)) as *const u32;
            if *id != bsp_apic_id {
                apic_ids.push(*id);
            }
        }

        // Ensure the BSP APIC ID was present
        assert_eq!(apic_ids.len(), *count as usize);
    }
    // 为什么va可以用来表示apic_ids的count数，还需要进一步认证
    // 需要完全搞清楚GHCB_NAE_GET_APIC_IDS这个协议究竟做了什么，不过我现在看懂了差不多
    pgtable_make_pages_private(va, pages * PAGE_SIZE);
    mem_free_frames(frame, pages);

    apic_ids
}
```
这个函数的解读存在一定的难度，我们暂且对此有所保留，不过确实不太妨碍我们将`__percpu_init`函数顺利解读完。


到这里，我们对于percpu_init的函数的解读也就完成了，总结一下就是找到可利用的全部`apic_id`总数，并为他们分配用于存储自身相关特征的内存空间，做好基本的初始化工作。

当然，也有两个历史遗留问题没有解决：
1. 为什么可以通过`va`的数值来确定`APs`的总数
2. `GHCB_NAE_GET_APIC_IDS`协议在`GHCB`中的具体行为表现，或许我可以去代码仓库那边问一问情况