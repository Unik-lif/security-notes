## 关于`start_bios`源码的分析
```rust
/// Locate BIOS, prepare it, advertise SVSM presence and run BIOS
pub fn start_bios() {
    prepare_bios();

    if !smp_run_bios_vmpl() {
        vc_terminate_svsm_general();
    }
}

fn prepare_bios() {
    // 获取bios_pa和bios_size的值
    let (bios_pa, bios_size) = match fwcfg_get_bios_area() {
        Some(r) => r,
        None => vc_terminate_svsm_fwcfg(),
    };
    // 利用bios_pa和bios_size来获取bios_map
    // 这个bios_map以MapGuard来进行包装
    let bios_map: MapGuard = match MapGuard::new_private(bios_pa, bios_size) {
        Ok(m) => m,
        Err(_e) => vc_terminate_svsm_fwcfg(),
    };
    // 利用bios_map建立bios_info
    let mut bios_info: BiosInfo = BiosInfo::new(bios_map.va(), bios_size);
    // 设置guid_table，根据bios_info
    if !parse_bios_guid_table(&mut bios_info) {
        vc_terminate_svsm_bios();
    }
    // 根据bios_info来确定caa存放的物理地址
    let caa: PhysAddr = match locate_bios_ca_page(&mut bios_info) {
        Some(p) => p,
        None => vc_terminate_svsm_bios(),
    };

    unsafe {
        // 这个函数看样子非常重要，我们需要仔细解读
        // 不过在此之前，我们需要对启动的流程重新产生一些理解
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
    // 走FW_CFG_SELECTOR的端口
    select_cfg_item(selector);
    // 把返回的数据读取出来，此即为BIOS_GPA
    bios_pa = read64_data_le();

    let selector: u16 = match find_file_selector(FW_CFG_BIOS_SIZE) {
        Some(f) => f,
        None => return None,
    };
    select_cfg_item(selector);
    bios_size = read64_data_le();

    // Check for possible buffer overflow
    match bios_pa.checked_add(bios_size) {
        Some(_v) => (),
        None => return None,
    };

    Some((PhysAddr::new(bios_pa), bios_size))
}

// 根据fname来确定到底找哪些配置文件
fn find_file_selector(fname: &str) -> Option<u16> {
    // FW_CFGS_FILES是我们通过DMA读已经获取的配置文件集合
    let files: LockGuard<Vec<FwCfgFile>> = FW_CFG_FILES.lock();

    // 对于这一部分文件进行遍历
    for f in files.iter() {
        // 找到和fname名字所match的文件
        // 先找到0，表示有多长
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