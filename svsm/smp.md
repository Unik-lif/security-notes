## smp_init函数的解读
先前我们的操作均是在BSP的角度来看问题，我们在`PerCpu_Init`相关函数中，已经实现了`APs`的建立，下一步便是启动这些`AP`核。
```rust
/// Boot other CPUs (APs)
pub fn smp_init() {
    unsafe {
        __smp_init();
    }
}

unsafe fn __smp_init() {
    // 这些参数来自于start.S中的特定物理位置
    // 很不凑巧，hl_main位置恰好是svsm_main位置，即BSP的启动位置
    set_hl_main(ap_entry as u64);
    // 设置cpu_mode为1，不知道目前有什么用
    set_cpu_mode(1);

    let count: usize = percpu_count();
    let aux: usize = count - 1;

    prints!("> Starting SMP for {aux} APs:\n");

    for i in 1..count {
        if !ap_start(i) {
            vc_terminate_svsm_general();
        }
        prints!("-- AP {i}/{aux} initialized.\n");
    }
}
```
我们看到`AP`核的启动位置被写为`ap_entry`，之后再利用一个简单的循环将除了`BSP`以外的核，即`AP`核进行启动，对此进行研究需要观察`ap_entry`和`ap_start`两个函数的具体情况。

### 相关函数解析：
#### ap_entry
```rust
/// Function executed for each AP when booted
pub extern "C" fn ap_entry() -> ! {
    unsafe {
        // ap核所对应的ghcb在先前确实没有被注册过
        // 先前被注册拿去使用的只是BSP相关的GHCB
        // 这一部分GHCB信息仅仅只是分配了相关的内存空间
        // 为了在之后拿去使用，需要进行注册
        vc_register_ghcb(pgtable_va_to_pa(PERCPU.ghcb()));
        // 设置Barrier保证序列化
        // 这个函数展开来如下所示：
        // Make sure threads are sequentially consistent
        //  #[macro_export]
        //  macro_rules! BARRIER {
        //      () => {
        //          core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst)
        //          };
        //  }
        // 效果是让线程能够序列化，按照Ordering::SeqCst中的给定顺序来运行
        BARRIER!();
        // 通过了这个Barrier，不出意外的话同步已经完成了
        // 这个时候AP_SYNC的状态为AP_STARTED，表示AP确实已经启动了
        AP_SYNC = AP_STARTED;
    }
    // 汇编hlt指令，仅此而已
    halt();
    // svsm_request_loop是个较为核心的函数
    svsm_request_loop();

    loop {
        halt()
    }
}
```
该函数的大体功能和注意选项我们已经在注释中讲的比较明白了，现在我们尝试对`svsm_request_loop`函数做一下解读。
#### svsm_request_loop:
```rust
/// Process SVSM requests
pub fn svsm_request_loop() {
    loop {
        match process_one_request(VMPL::Vmpl1) {
            Ok(()) => (),
            Err(e) => prints!("{}", e),
        };
        vc_run_vmpl(VMPL::Vmpl1);
    }
}
```
看样子这是一个大循环，最终目的反正是希望能够让`AP`以`VMPL::Vmpl1`的权限级来启动，我们来看这是怎么做到的。

这个函数本质上是在`vmpl`级别下进行一个请求的处理流程。
```rust
fn process_one_request(vmpl: VMPL) -> Result<(), String> {
    //
    // Limit the mapping of guest memory to only what is needed to process
    // the request.
    //
    // caa: calling area address
    // 获取这个部分的物理地址
    // 比较抽象的是，现在我们还似乎在BSP的PERCPU上做工作
    let caa_gpa: PhysAddr = unsafe { PERCPU.caa(vmpl) };
    let mut ca_map: MapGuard = match MapGuard::new_private(caa_gpa, CAA_MAP_SIZE) {
        Ok(m) => m,
        Err(e) => return Err(alloc::format!("Error mapping guest calling area: {e:?}")),
    };

    let vmsa_gpa: PhysAddr = unsafe { PERCPU.vmsa(vmpl) };
    let mut vmsa_map: MapGuard = match MapGuard::new_private(vmsa_gpa, VMSA_MAP_SIZE) {
        Ok(m) => m,
        Err(e) => return Err(alloc::format!("Error mapping guest VMSA: {e:?}")),
    };

    // 总之上面找到了caa_gpa和vmsa_gpa，然后给他们分配对应的空间
    // 不过是不是BSP的，这点我不好说
    // 之后利用函数vmsa_clear_efer_svme来跑
    // 在特权级下，清空了EFER寄存器SVME bit位
    // 正确执行会返回true值
    if !vmsa_clear_efer_svme(vmsa_map.va()) {
        let msg: String = alloc::format!("map_guest_input: vmsa_clear_efer_svme() failed");
        return Err(msg);
    }

    let vmsa: &mut Vmsa = vmsa_map.as_object_mut();
    let ca: &mut Ca = ca_map.as_object_mut();
    // 根据BSP对应的PERCPU来确定
    // guest_exitcode以及call_pending是否按照下面的方式进行设置
    if vmsa.guest_exitcode() == VMEXIT_VMGEXIT && ca.call_pending() == 1 {
        // 如果是上面这么设置的，说明确实是有调用的请求
        // 每次处理一个请求
        unsafe { handle_request(&mut *vmsa) };
        ca.set_call_pending(0);
    }

    //
    // Set EFER.SVME to 1 to allow the VMSA to be run by the hypervisor.
    //
    vmsa_set_efer_svme(vmsa_map.va());

    Ok(())
}
```
在这边有一个新的数据结构名为`caa`，这个数据结构是自`SVSM`发布之后新添加的一类特性。其主要的功能是反映用户对于`SVSM`相关的功能调用情况，以及当前是否存在着空闲的内存等信息。

原文把这个东西写的很好：

> To make a call to the SVSM, the guest OS must load the RAX register with the identifier of the call, where bits [63:32] hold the protocol number and bits [31:0] hold the call identifier within the protocol. Additional registers and/or memory may need to be configured with values specific to the call being issued. Once all memory and registers have been prepared, the guest OS must write a value of 1 to the SVSM_CALL_PENDING field of the Calling Area to indicate its readiness to issue the call. Finally, the guest OS must execute VMGEXIT to request that the host execute the SVSM on behalf of the calling vCPU. This request is made in one of two ways, either by using the GHCB MSR protocol with a value of 0x16 or by setting GHCB fields SW_EXITCODE=0x8000_0017 and SW_EXITINFO1=0.(See the GHCB Specification for furtherdetails.)

然而截止2023年7月25日，这个东西并没有更新，所以我们只好通过代码来猜猜看了。

针对`MapGuard`数据结构，我们有以下的代码：
```rust
pub struct MapGuard {
    pa: PhysAddr,
    va: VirtAddr,
    len: u64,
    unmap_on_drop: bool,
}

impl MapGuard {
    /// Map an area to virtual memory as private (encrypted) pages; when
    /// the MapGuard is dropped, the area will be unmapped.
    pub fn new_private(pa: PhysAddr, len: u64) -> Result<Self, MapToError<Size4KiB>> {
        // 直接调用Map相关的函数，可以执行最纯粹的映射工作
        let va: VirtAddr = pgtable_map_pages_private(pa, len)?;
        Ok(Self {
            pa,
            va,
            len,
            unmap_on_drop: true,
        })
    }
}


impl Drop for MapGuard {
    fn drop(&mut self) {
        if self.unmap_on_drop {
            pgtable_unmap_pages(self.va, self.len);
        }
    }
}
```
创建一个`MapGuard`类型的空间，在`unmap_on_drop`为`true`时，超出函数范围的时候他会自动`drop`掉。

不过对于`MapGuard`在`len`仅仅`8`个字节的情况下确实还是有那么一些些浪费哈哈哈，或许这是一个可以考虑被改进的地方？又或许其实它用了`SLAB`但是我没有发现？

函数`vmsa_clear_efer_svme`的解读如下：输入的`va`是`vmsa_map`的`va`地址。我们通过阅读手册可以得知，`SVME`bit是`EFER`寄存器中掌管虚拟化的一个关键bit位，只有在它被设置为`1`时，`SVM`的相关功能才能被启用。

这个`EFER`寄存器非常重要，是特权级才能动的寄存器，以至于`Guest OS`是绝对不能动它的，`Hypervisor`需要避免`Guest OS`去访问它。
```rust
unsafe fn update_vmsa_efer_svme(va: VirtAddr, svme: bool) -> bool {
    flush(va);
    // 同步
    BARRIER!();

    let vmsa: *mut Vmsa = va.as_mut_ptr();
    let efer_va: u64 = va.as_u64() + (*vmsa).efer_offset();

    let cur_efer: u64 = (*vmsa).efer();
    // svme -> bool type.
    // 添加，或者去除相关的位
    let new_efer: u64 = match svme {
        true => cur_efer | EFER_SVME,
        false => cur_efer & !EFER_SVME,
    };
    // x86架构特有的xchg操作，可以当做原子（反正用lock来做了，
    // 这玩意儿就是个0-1零和博弈，要么不做，要么做完
    // 反正就是为了做好同步化数值比较用的

    // cur_efer和new_efer是否相等，相等则替换efer_va指向的内容为new_efer
    let xchg_efer: u64 = cmpxchg(cur_efer, new_efer, efer_va);
    BARRIER!();

    // If the cmpxchg() succeeds, xchg_efer will have the cur_efer value,
    // otherwise, it will have the new_efer value.
    xchg_efer == cur_efer
}

pub fn vmsa_clear_efer_svme(va: VirtAddr) -> bool {
    unsafe { update_vmsa_efer_svme(va, false) }
}

/// Compare and exchange
// va对应0，newval对应1
// 首先检查rax寄存器，即cur_efer和new_efer是否是相同的，如果是相同的
// va指向的东西的值会被替换为newval，同时xchg_efer的值会和cur_efer一样，都是rax的值
pub fn cmpxchg(cmpval: u64, newval: u64, va: u64) -> u64 {
    let ret: u64;

    unsafe {
        asm!("lock cmpxchg [{0}], {1}",
             in(reg) va, in(reg) newval, in("rax") cmpval,
             lateout("rax") ret,
             options(nostack));
    }

    ret
}
```
#### handle_request函数
```rust
unsafe fn handle_request(vmsa: *mut Vmsa) {
    let protocol: u32 = UPPER_32BITS!((*vmsa).rax()) as u32;
    let callid: u32 = LOWER_32BITS!((*vmsa).rax()) as u32;

    match protocol {
        SVSM_CORE_PROTOCOL => core_handle_request(callid, vmsa),
        _ => (*vmsa).set_rax(SVSM_ERR_UNSUPPORTED_PROTOCOL),
    }
}
```
该函数用于处理`request`，根据SVSM Draft，请求的写法还是比较有意思的。如果rax的高半截是0，那么说明对应的是成功的情况，最后得到的处理结果如下所示：
```rust
pub unsafe fn core_handle_request(callid: u32, vmsa: *mut Vmsa) {
    match callid {
        SVSM_CORE_QUERY_PROTOCOL => handle_query_protocol_request(vmsa),
        SVSM_CORE_REMAP_CA => handle_remap_ca_request(vmsa),
        SVSM_CORE_PVALIDATE => handle_pvalidate_request(vmsa),
        SVSM_CORE_CREATE_VCPU => handle_create_vcpu_request(vmsa),
        SVSM_CORE_DELETE_VCPU => handle_delete_vcpu_request(vmsa),
        SVSM_CORE_CONFIGURE_VTOM => handle_configure_vtom_request(vmsa),

        _ => (*vmsa).set_rax(SVSM_ERR_UNSUPPORTED_CALLID),
    };
}
```
我们看到在这里就将`SVSM`相关的服务串联起来了，可以通过`callid`和`vmsa`具体的值来确定究竟调用的是哪一类服务，这个信息可以从`SVSM_DRAFT`上获得更多的了解。

这些服务具体是什么，我们留给之后的解读工作。
#### vc_run_vmpl函数
下一个函数如下所示：
```rust
/// Each vCPU has two VMSAs: One for VMPL0 (for SVSM) and one for VMPL1 (for
/// the guest).
///
/// The SVSM will use this function to invoke a GHCB NAE event to go back to
/// the guest after handling a request.
///
/// The guest will use the same GHCB NAE event to request something of the SVSM.
///
pub fn vc_run_vmpl(vmpl: VMPL) {
    let ghcb: *mut Ghcb = vc_get_ghcb();

    unsafe {
        vc_perform_vmgexit(ghcb, GHCB_NAE_RUN_VMPL, vmpl as u64, 0);

        (*ghcb).clear();
    }
}
```
挺简单一个函数，说白了就是从`VMPL0`退出，跑路到`VMPL1`，具体参数不知道怎么设置，总之要通过`ghcb`来进行参数传递，然后进行一个`world switch`就完事儿了了。

#### 函数ap_start
这也是一个重量级函数。说实在的还好今年五月认真做完了一遍`rCore`并且以强迫症的态度对全部代码做解析，不然这个项目我根本没法跟下来，不仅仅是能力上的问题，可能更多是勇气上的问题。
```rust
/// Start a given AP, which includes creating a Stack and Vmsa
fn ap_start(cpu_id: usize) -> bool {
    // 以PERCPU作为基地址，找到apic_id的信息
    let apic_id: u32 = unsafe { PERCPU.apic_id_for(cpu_id) };
    // 为cpu_id所对应的这个AP分配了一个svsm_vmsa，运行在VMPL0之上
    let vmsa: VirtAddr = create_svsm_vmsa(cpu_id);
    // 用rmpadjust指令来处理vmsa对应的页，在attrs中写入对应的信息
    let ret: u32 = rmpadjust(vmsa.as_u64(), RMP_4K, VMSA_PAGE | VMPL::Vmpl1 as u64);
    if ret != 0 {
        vc_terminate_svsm_general();
    }
    // 为APs建立栈空间
    let stack: VirtAddr = mem_create_stack(SVSM_STACK_PAGES, false);
    set_cpu_stack(stack.as_u64());

    unsafe {
        // 为APs在VMPL0的情况下找到对应的vmsa
        PERCPU.set_vmsa_for(pgtable_va_to_pa(vmsa), VMPL::Vmpl0, cpu_id);
        // 原神，启动！！
        AP_SYNC = AP_STARTING;
        BARRIER!();
        // 通过virtual communication来真实创建ap
        vc_ap_create(vmsa, apic_id);

        while AP_SYNC != AP_STARTED {
            pause();
        }
    }

    true
}
```
从注释上来看它确实功能很简单，给定一个`AP`，让它跑起来，说白了就这么多。为了要跑起来应该怎么做？当然要去设置它的栈空间和`VMSA`相关的东西。

对于函数`create_svsm_vmsa`，其整体结构如下所示：为`for_id`所对应的`AP`分配设置了一个运行在`VMPL0`之上的`VMSA`。
```rust
/// Create VMSA (execution context information) for an AP
fn create_svsm_vmsa(for_id: usize) -> VirtAddr {
    // 分配一个用于存放vmsa的地址空间
    let frame: PhysFrame = alloc_vmsa();
    let vmsa_va: VirtAddr = pgtable_pa_to_va(frame.start_address());
    let vmsa: *mut Vmsa = vmsa_va.as_mut_ptr();
    // 创建gdt，idt表，并为每个CPU的地址
    let gdtr: DescriptorTablePointer = sgdt();
    let idtr: DescriptorTablePointer = sidt();
    let gs: VirtAddr = percpu_address(for_id);
    let tss: VirtAddr = tss_init_for(for_id);

    unsafe {
        (*vmsa).set_cs_selector(get_gdt64_kernel_cs() as u16);
        (*vmsa).set_cs_rtype(SVSM_CS_TYPE);
        (*vmsa).set_cs_limit(SVSM_CS_LIMIT);
        (*vmsa).set_cs_base(SVSM_CS_BASE);

        (*vmsa).set_tr_selector(get_gdt64_tss() as u16);
        (*vmsa).set_tr_rtype(SVSM_TSS_TYPE);
        (*vmsa).set_tr_limit(size_of::<TaskStateSegment>() as u32 - 1);
        (*vmsa).set_tr_base(tss.as_u64());

        (*vmsa).set_gs_base(gs.as_u64());

        (*vmsa).set_rip(get_cpu_start());

        (*vmsa).set_gdtr_limit(gdtr.limit as u32);
        (*vmsa).set_gdtr_base(gdtr.base.as_u64());

        (*vmsa).set_idtr_limit(idtr.limit as u32);
        (*vmsa).set_idtr_base(idtr.base.as_u64());

        (*vmsa).set_cr0(SVSM_CR0);
        (*vmsa).set_cr3(Cr3::read().0.start_address().as_u64());
        (*vmsa).set_cr4(SVSM_CR4);
        (*vmsa).set_efer(SVSM_EFER);
        (*vmsa).set_rflags(SVSM_RFLAGS);
        (*vmsa).set_dr6(SVSM_DR6);
        (*vmsa).set_dr7(SVSM_DR7);
        (*vmsa).set_gpat(SVSM_GPAT);
        (*vmsa).set_xcr0(SVSM_XCR0);
        (*vmsa).set_mxcsr(SVSM_MXCSR);
        (*vmsa).set_x87_ftw(SVSM_X87_FTW);
        (*vmsa).set_x87_fcw(SVSM_X87_FCW);

        (*vmsa).set_vmpl(VMPL::Vmpl0 as u8);
        (*vmsa).set_sev_features(rdmsr(MSR_SEV_STATUS) >> 2);
    }

    vmsa_va
}
```

最后一个函数如下所示：
```rust
pub fn vc_ap_create(vmsa_va: VirtAddr, apic_id: u32) {
    let ghcb: *mut Ghcb = vc_get_ghcb();
    let vmsa: *const Vmsa = vmsa_va.as_u64() as *const Vmsa;

    unsafe {
        let info1: u64 =
            GHCB_NAE_SNP_AP_CREATION_REQ!(SNP_AP_CREATE_IMMEDIATE, (*vmsa).vmpl(), apic_id);
        let info2: u64 = pgtable_va_to_pa(vmsa_va).as_u64();

        (*ghcb).set_rax((*vmsa).sev_features());

        vc_perform_vmgexit(ghcb, GHCB_NAE_SNP_AP_CREATION, info1, info2);

        (*ghcb).clear();
    }
}
```
在一切准备就绪的情况下，需要通过`Virtual Communication`来让`hypervisor`所知道需要处理这些要求。

至此，该函数的解读任务便结束了。