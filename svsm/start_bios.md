## 关于`start_bios`源码的分析
```rust
/// Locate BIOS, prepare it, advertise SVSM presence and run BIOS
pub fn start_bios() {
    prepare_bios();

    if !smp_run_bios_vmpl() {
        vc_terminate_svsm_general();
    }
}
```
其中利用了`prepare_bios`函数来做启动前的相关准备，而`smp_run_bios_vmpl`则将会以`vmpl1`等其它权限级来运行`AP`核。
```rust
fn prepare_bios() {
    // 获取bios_pa和bios_size的值
    // 通过配置文件找到选择子，然后利用选择子写端口，再通过读端口来获得最终的信息
    let (bios_pa, bios_size) = match fwcfg_get_bios_area() {
        Some(r) => r,
        None => vc_terminate_svsm_fwcfg(),
    };
    // 利用bios_pa和bios_size来获取bios_map
    // 这个bios_map以MapGuard来进行包装，MapGuard与此同时会构建一个向虚拟内存页的private类型映射
    let bios_map: MapGuard = match MapGuard::new_private(bios_pa, bios_size) {
        Ok(m) => m,
        Err(_e) => vc_terminate_svsm_fwcfg(),
    };
    // 利用bios_map建立bios_info，guid table此时也是new完的状态
    let mut bios_info: BiosInfo = BiosInfo::new(bios_map.va(), bios_size);
    // 设置guid_table，根据bios_info的一些已有信息，对bios_info.guid_table做好设置
    // 同时涉及一部分验证操作
    if !parse_bios_guid_table(&mut bios_info) {
        vc_terminate_svsm_bios();
    }
    // 根据bios_info来确定caa存放的物理地址，通过SNPSection的寻找来确定，这个流程其实还是比较复杂的
    let caa: PhysAddr = match locate_bios_ca_page(&mut bios_info) {
        Some(p) => p,
        None => vc_terminate_svsm_bios(),
    };

    unsafe {
        // 这个函数看样子非常重要，我们需要仔细解读
        // 不过在此之前，我们需要对启动的流程重新产生一些理解
        // 发现这里其实也不是那么要紧，并不是很重要
        if !advertise_svsm_presence(&mut bios_info, caa) {
            vc_terminate_svsm_bios();
        }
    }

    // 这里涉及了VMPL1级权限的操作，值得探讨
    if !smp_prepare_bios_vmpl(caa) {
        vc_terminate_svsm_general();
    }
}
```
`prepare_bios`函数首先通过`fwcfg_get_bios_area`函数来获得`bios_pa`，`bios_size`等`bios`相关信息，该函数具体如下所示。
```rust
/// Returns the GPA and size of the area in which the bios is loaded
pub fn fwcfg_get_bios_area() -> Option<(PhysAddr, u64)> {
    let bios_pa: u64;
    let bios_size: u64;
    // find_file_selector
    let selector: u16 = match find_file_selector(FW_CFG_BIOS_GPA) {
        Some(f) => f,
        None => return None,
    };
    // 获得的是fname文件名对应的选择子selector
    // 走FW_CFG_SELECTOR的端口，将selector作为item来写入
    select_cfg_item(selector);
    // 把返回的数据读取出来，此即为BIOS_GPA
    bios_pa = read64_data_le();

    // 如法炮制，这次希望获取的是FW_CFG_BIOS_SIZE，自然我们也可以通过新的选择子来获取bios_size
    let selector: u16 = match find_file_selector(FW_CFG_BIOS_SIZE) {
        Some(f) => f,
        None => return None,
    };
    select_cfg_item(selector);
    bios_size = read64_data_le();

    // Check for possible buffer overflow
    // 检查是否会越界
    match bios_pa.checked_add(bios_size) {
        Some(_v) => (),
        None => return None,
    };

    Some((PhysAddr::new(bios_pa), bios_size))
}
```
其中核心函数`find_file_selector`是强依赖，我们简单看一下这个函数做了什么事情：
```rust
// 根据fname来确定到底找哪些配置文件
// 我们之前把相关的Firmware Configuration配置文件存放到了FW_CFG_FILES之中，现在从中直接获取它就可以了
fn find_file_selector(fname: &str) -> Option<u16> {
    // FW_CFGS_FILES是我们通过DMA读已经获取的配置文件集合
    let files: LockGuard<Vec<FwCfgFile>> = FW_CFG_FILES.lock();

    // 对于这一部分文件进行遍历
    for f in files.iter() {
        // 找到和fname名字所match的文件
        // 先找到f.name之中的'\0'，表示字符串到底有多长
        let nul: usize = match memchr(0, &f.name) {
            Some(n) => n,
            None => vc_terminate_svsm_fwcfg(),
        };
        // 获取长度为nul的字符串
        let n: &str = match core::str::from_utf8(&f.name[0..nul]) {
            Ok(n) => n,
            Err(_e) => vc_terminate_svsm_fwcfg(),
        };
        // 需要和fname完全一致，才能提前返回
        if n.eq(fname) {
            return Some(f.select);
        }
    }

    return None;
}
```
因此本质上上面这个流程是通过`fname`找到对应的`f.select`的流程。

之后，利用`parse_bios_guild_table`函数实现`guid_table`的设置工作。
```rust
fn parse_bios_guid_table(bios_info: &mut BiosInfo) -> bool {
    if bios_info.size() < (BIOS_TABLE_END + GUID_SIZE + BIOS_TABLE_LEN_FIELD) {
        return false;
    }

    unsafe {
        let bios: *const u8 = (bios_info.va() + bios_info.size() - BIOS_TABLE_END) as *const u8;
        let bytes: *const Bytes = (bios as u64 - GUID_SIZE) as *const Bytes;

        let guid: Uuid = Uuid::from_bytes_le(*bytes);
        if guid != OVMF_TABLE_GUID {
            return false;
        }

        let len: *const u16 = (bios as u64 - GUID_SIZE - BIOS_TABLE_LEN_FIELD) as *const u16;
        if (*len as u64) > bios_info.size() {
            return false;
        }

        bios_info.guid_table.set_begin(bios as u64 - *len as u64);
        bios_info.guid_table.set_end(len as u64);
        bios_info.guid_table.set_len(*len);
    }

    true
}
```
再者，利用`locate_bios_ca_page`函数，根据`bios_info`信息，找到对应的`SnpSection`位置，返回它的物理地址。
```rust
fn locate_bios_ca_page(bios_info: &mut BiosInfo) -> Option<PhysAddr> {
    // 找到SNP中存放SVSM_CAA的部分，即snp_section中stype与之相同的SnpSection
    let section: SnpSection = match find_snp_section(bios_info, SNP_SECT_SVSM_CAA) {
        Some(p) => p,
        None => return None,
    };

    if (section.size() as usize) < size_of::<u32>() {
        return None;
    }
    // 返回bios_ca_page所在的物理地址
    return Some(PhysAddr::new(section.address_u64()));
}
```
其中有个重要函数是`find_snp_section`，我们尝试简单研究一下：
```rust
fn find_snp_section(bios_info: &mut BiosInfo, stype: u32) -> Option<SnpSection> {
    // 在guid之中遍历，如果返回是Some，说明确实有OVMF_SNP_ENTRY_GUID存在
    // 同时p一定程度上能够反映其在guid表中的偏移量，可以辅助函数__find_snp_section来运作
    let p: u64 = match find_bios_guid_entry(bios_info, OVMF_SNP_ENTRY_GUID) {
        Some(p) => p,
        None => vc_terminate_svsm_bios(),
    };

    unsafe { __find_snp_section(bios_info, stype, p) }
}

fn find_bios_guid_entry(bios_info: &mut BiosInfo, target_guid: Uuid) -> Option<u64> {
    let mut avail_len: u64 = bios_info.guid_table.len() as u64;
    let mut p: u64 = bios_info.guid_table.end();
    // 根据bios_info和target_guid继续往下找，具体来说要使用guid之中的信息
    unsafe { __find_bios_guid_entry(bios_info, target_guid, &mut avail_len, &mut p) }
}

unsafe fn __find_bios_guid_entry(
    bios_info: &mut BiosInfo,
    target_guid: Uuid,
    avail_len: &mut u64,
    p: &mut u64,
) -> Option<u64> {
    /* Search is in reverse order */
    // 从guid的尾端逐级相减，尝试找到entry_guid与target_guid相符合的值
    while *p > bios_info.guid_table.begin() {
        let len: u64 = *((*p - GUID_SIZE - BIOS_TABLE_LEN_FIELD) as *const u16) as u64;
        if (len < (GUID_SIZE + BIOS_TABLE_LEN_FIELD)) || (len > *avail_len) {
            return None;
        }

        let bytes: *const Bytes = (*p - GUID_SIZE) as *const Bytes;
        let entry_guid: Uuid = Uuid::from_bytes_le(*bytes);
        if entry_guid == target_guid {
            return Some(*p - len as u64);
        }

        *avail_len -= len;
        *p -= len;
    }

    return None;
}
```
在获得了guid相关的偏移量后，我们可以进入函数`__find_snp_section`之中研究了。
```rust
unsafe fn __find_snp_section(bios_info: &mut BiosInfo, stype: u32, p: u64) -> Option<SnpSection> {
    let offset: u64 = ptr::read_unaligned(p as *const u32) as u64;
    if offset > bios_info.size() {
        return None;
    }
    // 利用offset索引到metadata，即SnpMeteData位置
    // 这个数据结构存放着签名、版本等有关信息
    let metadata: *const SnpMetaData =
        (bios_info.va() + bios_info.size() - offset) as *const SnpMetaData;
    if (*metadata).signature() != SNP_METADATA_SIGNATURE {
        return None;
    }
    // 将defined_len和expected_len计算出来比较
    let defined_len: u64 = (*metadata).len() as u64;
    let expected_len: u64 = (*metadata).section_count() as u64 * size_of::<SnpSection>() as u64;
    if defined_len < expected_len {
        return None;
    }
    // 利用SnpMetedata通过地址偏移量找到SnpSection的地址
    let mut section: *const SnpSection =
        (metadata as u64 + size_of::<SnpMetaData>() as u64) as *const SnpSection;
    for _i in 0..(*metadata).section_count() {
        // 遍历全部的SnpSection，直到stype与我们找到的SnpSection相同
        if (*section).stype() == stype {
            return Some(*section);
        }

        section = (section as u64 + size_of::<SnpSection>() as u64) as *const SnpSection;
    }

    return None;
}
```
下面我们分析一下很重要的一个函数`advertise_svsm_prensence`:本质上还是对`SnpSection`，`SnpSecrets`等启动依赖的验证项做一些设置。
```rust
unsafe fn advertise_svsm_presence(bios_info: &mut BiosInfo, caa: PhysAddr) -> bool {
    // 利用find_snp_section函数找到SNP_SECT_SECRETS所对应的SnpSection虚拟地址
    let section: SnpSection = match find_snp_section(bios_info, SNP_SECT_SECRETS) {
        Some(p) => p,
        None => return false,
    };
    
    if (section.size() as usize) < size_of::<SnpSecrets>() {
        return false;
    }
    // 找到section对应的物理地址
    let bios_secrets_pa: PhysAddr = PhysAddr::new(section.address_u64());
    let mut bios_secrets_map: MapGuard =
        match MapGuard::new_private(bios_secrets_pa, section.size_u64()) {
            Ok(m) => m,
            Err(_e) => return false,
        };
    let svsm_secrets_va: VirtAddr = get_svsm_secrets_page();

    // Copy the Secrets page to the BIOS Secrets page location
    // 让MapGuard以SnpSecrets的方式进行解析
    let bios_secrets: &mut SnpSecrets = bios_secrets_map.as_object_mut();
    let svsm_secrets: *const SnpSecrets = svsm_secrets_va.as_ptr();
    *bios_secrets = *svsm_secrets;

    // Clear the VMPCK0 key
    bios_secrets.clear_vmpck0();

    // Advertise the SVSM
    // SVSM所对应的bios_secrets看来是建立在VMPL1之上的
    bios_secrets.set_svsm_base(pgtable_va_to_pa(get_svsm_begin()).as_u64());
    bios_secrets.set_svsm_size(get_svsm_end().as_u64() - get_svsm_begin().as_u64());
    bios_secrets.set_svsm_caa(caa.as_u64());
    bios_secrets.set_svsm_max_version(1);
    bios_secrets.set_svsm_guest_vmpl(1);

    let section: SnpSection = match find_snp_section(bios_info, SNP_SECT_CPUID) {
        Some(p) => p,
        None => return false,
    };

    let bios_cpuid_pa: PhysAddr = PhysAddr::new(section.address_u64());
    let size: u64 = min(section.size_u64(), get_svsm_cpuid_page_size());

    let mut bios_cpuid_map: MapGuard = match MapGuard::new_private(bios_cpuid_pa, size) {
        Ok(m) => m,
        Err(_e) => return false,
    };
    let bios_cpuid: &mut [u8] = bios_cpuid_map.as_bytes_mut();

    let svsm_cpuid_va: VirtAddr = get_svsm_cpuid_page();
    let svsm_cpuid_ptr: *const u8 = svsm_cpuid_va.as_ptr();
    let svsm_cpuid: &[u8] = unsafe { slice::from_raw_parts(svsm_cpuid_ptr, size as usize) };

    // Copy the CPUID page to the BIOS Secrets page location
    bios_cpuid.copy_from_slice(svsm_cpuid);

    true
}
```
在完成了上述的繁琐设置之后，我们的`svsm`相关标签和`caa`信息都已经得到了较好的初始化，现在我们进入函数`smp_prepare_bios_vmpl`，尝试以其他特权级来运行`SVSM`。
```rust
/// Create a Vmsa and Caa and prepare them
pub fn smp_prepare_bios_vmpl(caa_pa: PhysAddr) -> bool {
    // 为BSP开辟了一片空间来存放vmsa信息
    let vmsa_pa: PhysAddr = alloc_vmsa().start_address();
    let vmsa: MapGuard = match MapGuard::new_private(vmsa_pa, VMSA_MAP_SIZE) {
        Ok(v) => v,
        Err(_e) => return false,
    };
    // 利用函数__create_bios_vmsa，从vmsa的起始虚拟地址开始执行
    // 现在为bsp创建了一个vmpl设置为vmpl1级别的vmsa
    // 现在需要在PERCPU中把这个东西设置在vmpl1位置处
    unsafe { __create_bios_vmsa(vmsa.va()) }

    let caa: MapGuard = match MapGuard::new_private(caa_pa, CAA_MAP_SIZE) {
        Ok(c) => c,
        Err(_e) => return false,
    };

    // 设置caa空间和刚刚利用create_bios_vmsa创建的vmsa于VMPL1之上
    unsafe {
        PERCPU.set_vmsa(vmsa_pa, VMPL::Vmpl1);
        PERCPU.set_caa(caa_pa, VMPL::Vmpl1);
    }

    // Update the permissions for the CAA and VMSA page.
    //
    // For the VMSA page, restrict it to read-only (at most) to prevent a guest
    // from attempting to alter the VMPL level within the VMSA.
    //
    // On error, do not try to reset the VMPL permission state for the pages,
    // just leak them.
    //
    // The lower VMPL has not been run, yet, so no TLB flushing is needed.
    //
    // 设置bsp对应的处在vmpl1的caa空间为在VMPL1权限中可以VMPL_RWX的类型
    let ret: u32 = rmpadjust(caa.va().as_u64(), RMP_4K, VMPL_RWX | VMPL::Vmpl1 as u64);
    if ret != 0 {
        return false;
    }
    // 设置vmsa为在VMPL1权限下仅仅可以去Read的类型
    let ret: u32 = rmpadjust(vmsa.va().as_u64(), RMP_4K, VMPL_R | VMPL::Vmpl1 as u64);
    if ret != 0 {
        return false;
    }

    let vmin: u64 = VMPL::Vmpl2 as u64;
    let vmax: u64 = VMPL::VmplMax as u64;
    // 对于其他的权限，仅用rmpadjust表面上过一遍，明确告知它们没有权限操作caa和vmsa空间
    for i in vmin..vmax {
        let ret: u32 = rmpadjust(caa.va().as_u64(), RMP_4K, i);
        if ret != 0 {
            return false;
        }

        let ret: u32 = rmpadjust(vmsa.va().as_u64(), RMP_4K, i);
        if ret != 0 {
            return false;
        }
    }
    // 除了READ以外，还告知vmsa.va开始的一块内存空间将会被用于当做VMSA
    // 参考手册中RMPADJUST指令信息
    let ret: u32 = rmpadjust(vmsa.va().as_u64(), RMP_4K, VMPL_VMSA | VMPL::Vmpl1 as u64);
    if ret != 0 {
        return false;
    }

    unsafe {
        // 利用GS寄存器，将当前vmsa_pa和apic_id当做元素，添加到VMSA_LIST之中进行管理
        // 特别的，BSP对应的PERCPU信息恰好是PERCPU，所以可以直接这么用偏移量来计算
        svsm_request_add_init_vmsa(vmsa_pa, PERCPU.apic_id());
    }

    true
}
```
看一下函数`__create_bios_vmsa`的大体样貌：
```rust
unsafe fn __create_bios_vmsa(vmsa_va: VirtAddr) {
    let bsp_page_va: VirtAddr = get_bios_vmsa_page();

    let vmsa: *mut Vmsa = vmsa_va.as_mut_ptr();
    let bsp_page: *const Vmsa = bsp_page_va.as_ptr();

    // Copy the measured BIOS BSP VMSA page
    // 直接将原来在start.S中data段所包含的bsp_page拷贝到我们新分配的vmsa之中
    // 因此要看bsp_page具体是怎么设置的
    // 然而在代码中并没有找到关于bsp_page的设置相关，猜测是固件上就已经满足了这些要求
    // 这就和读取snp_pages得到的bios_secret相关信息的原理一样
    // 最终的结果反正是：bsp_vmsa_page之中存放着vmpl1的vmsa信息，二者进行了地址拷贝，里头到底存放着什么东西我们不知道
    // 从其他函数的操作流程中印证了这一猜想
    *vmsa = *bsp_page;

    if (*vmsa).vmpl() != VMPL::Vmpl1 as u8 {
        vc_terminate_svsm_incorrect_vmpl();
    }

    // Check the SEV-SNP VMSA SEV features to make sure guest will
    // execute with supported SEV features. It is better to not fix
    // the SEV features ourselves, since this could indicate an issue
    // on the hypervisor side.

    if (*vmsa).sev_features() & VMPL1_REQUIRED_SEV_FEATS != VMPL1_REQUIRED_SEV_FEATS {
        vc_terminate_vmpl1_sev_features();
    }

    if (*vmsa).sev_features() & VMPL1_UNSUPPORTED_SEV_FEATS != 0 {
        vc_terminate_vmpl1_sev_features();
    }
}
```
在运行完函数`smp_prepare_bios_vmpl`之后，我们的`VMPL`相关的东东在`BSP`的层面至少已经完成初始化操作了，现在可以尝试进入函数`smp_run_bios_vmpl`，它如下所示：
```rust
/// Retrieve Vmpl1 Vmsa and start it
pub fn smp_run_bios_vmpl() -> bool {
    unsafe {
        // Retrieve VMPL1 VMSA and start it
        let vmsa_pa: PhysAddr = PERCPU.vmsa(VMPL::Vmpl1);
        if vmsa_pa == PhysAddr::zero() {
            return false;
        }

        let vmsa_map: MapGuard = match MapGuard::new_private(vmsa_pa, PAGE_SIZE) {
            Ok(r) => r,
            Err(_e) => return false,
        };

        vc_ap_create(vmsa_map.va(), PERCPU.apic_id());
    }

    true
}
```
简而言之，其从`BSP`所对应的`PERCPU`中获取了`Vmpl1`的物理地址，为它建立`MapGuard`且`private`结构，采用`vc_ap_create`函数做结尾，从而让`BSP`之后也能成为一个`AP`。
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
我们在上面看到了一个相对完整的建立`VMPL1`，修改其他权限级，获取`vmsa`页（通过`MapGuard`来做），以及通过`GHCB_NAE_SNP_AP_CREATION`协议建立`ap`的完整流程。

至此，本章的代码解读结束。