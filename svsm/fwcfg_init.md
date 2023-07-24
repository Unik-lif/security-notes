## 有关于函数fwcfg_init的分析
这个函数和`DMA`总线有一定的关联，也就是说，我确实不怎么会。

还是尝试简单看一下吧。
```rust
/// Perform DMA to read firmware configuration files
// fwcfg; firmware configuration.
pub fn fwcfg_init() {
    // 判断FwCfgDma的大小何如
    STATIC_ASSERT!(size_of::<FwCfgDma>() == PAGE_SIZE as usize);
    // 就只是非常简单的正常初始化过程而已
    lazy_static::initialize(&FW_CFG_DMA);

    /* Validate the signature */
    select_cfg_item(FW_CFG_SIGNATURE);
    let signature: u32 = read32_data_le();
    if signature != FW_SIGNATURE {
        vc_terminate_svsm_fwcfg();
    }

    /* Validate DMA support */
    select_cfg_item(FW_CFG_ID);
    let features: u32 = read32_data_le();
    if (features & FW_FEATURE_DMA) == 0 {
        vc_terminate_svsm_fwcfg();
    }

    select_cfg_item(FW_CFG_FILE_DIR);
    let file_count: u32 = read32_data_be();
    if file_count == 0 {
        vc_terminate_svsm_fwcfg();
    }

    unsafe {
        FILE_COUNT = file_count as usize;
    }

    lazy_static::initialize(&FW_CFG_FILES);

    unsafe {
        let f: FwCfgFile = FwCfgFile {
            size: 0,
            select: 0,
            reserved: 0,
            name: [0; 56],
        };

        let mut files: LockGuard<Vec<FwCfgFile>> = FW_CFG_FILES.lock();

        let size: usize = size_of::<FwCfgFile>();
        let mut control = FW_CFG_DMA_READ;
        let mut dma: LockGuard<&'static mut FwCfgDma> = FW_CFG_DMA.lock();
        for i in 0..FILE_COUNT {
            let bytes: *const u8 = &f as *const FwCfgFile as *const u8;
            perform_dma(&mut dma, bytes, control, size);

            files[i].size = u32::swap_bytes(f.size);
            files[i].select = u16::swap_bytes(f.select);
            files[i].name = f.name;

            /* Stay on the same item */
            control &= FW_CFG_DMA_CLEAR_SELECTOR;
        }

        prints!("> All {FILE_COUNT} firmware config files read.\n");
    }
}
```
## 基础数据结构与细节：
### FwCfgDma：
这个数据结构长下面这个样子：
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FwCfgDma {
    desc: FwCfgDmaDesc,
    data: [u8; DMA_DATA_SIZE],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FwCfgDmaDesc {
    control: u32,
    length: u32,
    address: u64,
}
```
其中的`FwCfgDmaDesc`数据结构含有控制、长度、地址共三个域，用来描述`Dma`的一些性质，而`DMA_DATA_SIZE`则反映了数据域，大小为`4080`字节，与`desc`并在一起正好是`4096`字节，符合我们代码上述提供的认知。
### FW_CFG_DMA:
我们来简单看一下下面的数据结构：
```rust
lazy_static! {
    // FW_CFG_DMA数据结构
    static ref FW_CFG_DMA: SpinLock<&'static mut FwCfgDma> = {
        // 首先分配一个物理frame
        let frame: PhysFrame = match mem_allocate_frame() {
            Some(f) => f,
            None => vc_terminate_svsm_enomem(),
        };
        // 找到物理页所对应的虚拟地址
        let va: VirtAddr = pgtable_pa_to_va(frame.start_address());

        // 让这一小部分的虚拟地址处于共享状态，空间大小为PAGE_SIZE
        pgtable_make_pages_shared(va, PAGE_SIZE);
        memset(va.as_mut_ptr(), 0, PAGE_SIZE as usize);

        let dma: &mut FwCfgDma;
        unsafe {
            dma = &mut *va.as_mut_ptr() as &mut FwCfgDma;
        }

        SpinLock::new(dma)
    };
    static ref FW_CFG_FILES: SpinLock<Vec<FwCfgFile>> = {
        let mut files: Vec<FwCfgFile>;

        unsafe {
            files = Vec::with_capacity(FILE_COUNT);
            for _i in 0..FILE_COUNT {
                let f: FwCfgFile = FwCfgFile::new();
                files.push(f);
            }
        }

        SpinLock::new(files)
    };
}
```
那其实确实针对`FW_CFG_DMA`的初始化很简单，分配一个大小和页一样的空间，然后以`FwCfgDma`的类型对他进行解读就完事儿了，非常简单。

之后用`SpinLock`进行管理。
### 函数们：
#### vc_outw:
该函数被`select_cfg_item`所调用，具体内容如下所示：

其中`port`对应的值是`FW_CFG_SELECTOR`，而`value`对应的值是`FW_CFG_SIGNATURE`，因此ioio的信息具体是什么我们就可以知晓一部分。
```rust
pub fn vc_outw(port: u16, value: u16) {
    let ghcb: *mut Ghcb = vc_get_ghcb();
    let mut ioio: u64 = (port as u64) << 16;

    ioio |= IOIO_ADDR_64;
    ioio |= IOIO_SIZE_16;

    unsafe {
        (*ghcb).set_rax(value as u64);

        // 这边对应的协议请同样参考GHCB Specification
        vc_perform_vmgexit(ghcb, GHCB_NAE_IOIO, ioio, 0);

        (*ghcb).clear();
    }
}
```