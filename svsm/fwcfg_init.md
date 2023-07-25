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
    // 验证签名是否合法，通过intercept某特定端口的方法来实现
    // 对应的特殊协议为GHCB_NAE_IOIO
    // ---------------------标准流程------------------------
    // 先通过把数据从Guest中写入到Hypervisor内
    // 需要out函数来操作
    // 数据将会输出到指定的端口上
    // 再通过IN方式把数据从Hypervisor内读取到Guest中
    // 除了DMA可以自动读取信息到对应内存中，其他的均需要用IN与之进行匹配
    // ----------------------------------------------------
    select_cfg_item(FW_CFG_SIGNATURE);
    let signature: u32 = read32_data_le();
    if signature != FW_SIGNATURE {
        vc_terminate_svsm_fwcfg();
    }

    /* Validate DMA support */
    // 与前面的代码结构完全一致
    select_cfg_item(FW_CFG_ID);
    let features: u32 = read32_data_le();
    if (features & FW_FEATURE_DMA) == 0 {
        vc_terminate_svsm_fwcfg();
    }

    // 去看一下有多少个file
    // 不过这边用的好像是大端序
    select_cfg_item(FW_CFG_FILE_DIR);
    let file_count: u32 = read32_data_be();
    if file_count == 0 {
        vc_terminate_svsm_fwcfg();
    }

    unsafe {
        FILE_COUNT = file_count as usize;
    }

    // 在这一步已经把需要用到的空间全部分配好了，接下来只要做到填充就可以了。
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
        // 对这么多个文件进行DMA上的阅读
        for i in 0..FILE_COUNT {
            // f是一个类似于temp寄存器一样的存在
            let bytes: *const u8 = &f as *const FwCfgFile as *const u8;
            // 通过函数perform_dma来实现DMA工作
            // 东西将会存放在bytes之中，我们之后以合适的方式进行解读就行了。
            // 部分swap_bytes可能与外设和内存不同的读取顺序有关系
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
}
```
那其实确实针对`FW_CFG_DMA`的初始化很简单，分配一个大小和页一样的空间，然后以`FwCfgDma`的类型对他进行解读就完事儿了，非常简单。

之后用`SpinLock`进行管理。
### FW_CFG_FILES
```rust
lazy_static! {
    // FW_CFG_FILES数据结构
    static ref FW_CFG_FILES: SpinLock<Vec<FwCfgFile>> = {
        let mut files: Vec<FwCfgFile>;

        unsafe {
            // 根据FILE_COUNT中的值来确定files空间，并为其添加FwcfgFile的初始项
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
`FW_CFG_FILES`通过`FILE_COUNT`来确定最终的存放`File`数目，与此同时，`FwCfgFile`的数据类型如下所示：
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FwCfgFile {
    size: u32,
    select: u16,
    reserved: u16,
    name: [u8; 56],
}

#[allow(dead_code)]
impl FwCfgFile {
    pub const fn new() -> Self {
        FwCfgFile {
            size: 0,
            select: 0,
            reserved: 0,
            name: [0; 56],
        }
    }
    funcs!(size, u32);
    funcs!(select, u16);
    funcs!(name, [u8; 56]);
}
```
专门用来读取`Firmware Configuration`信息。
### 函数们：
#### vc_outw:
该函数被`select_cfg_item`所调用，具体内容如下所示：

其中`port`对应的值是`FW_CFG_SELECTOR`，而`value`对应的值是`FW_CFG_SIGNATURE`，因此ioio的信息具体是什么我们就可以知晓一部分。
```rust
pub fn vc_outw(port: u16, value: u16) {
    // 仍然是对BSP上的处理
    // 得到PERCPU上的GHCB位置
    let ghcb: *mut Ghcb = vc_get_ghcb();
    // port被设置成了需要被切断输入输出的FW_CFG_SELECTOR
    let mut ioio: u64 = (port as u64) << 16;
    // 在9和5上写入了位置
    // 表示64位地址和16位的操作数
    // type位设置为了0，表示是OUT，即仅仅把东西写入，不返回什么值
    // 如果有问题在vc_perform_vmgexit之中就会出现相关的报错
    ioio |= IOIO_ADDR_64;
    ioio |= IOIO_SIZE_16;

    unsafe {
        // rax需要被设置成value，即signature
        (*ghcb).set_rax(value as u64);

        // 这边对应的协议请同样参考GHCB Specification
        // 处理IO fault的标准方式，用来执行总线的读写工作
        // 自然，也需要通知hypervisor
        vc_perform_vmgexit(ghcb, GHCB_NAE_IOIO, ioio, 0);

        (*ghcb).clear();
    }
}
```
#### read32_data_le:
```rust
fn read32_data_le() -> u32 {
    let mut value: u32 = vc_inb(FW_CFG_DATA) as u32;

    value |= (vc_inb(FW_CFG_DATA) as u32) << 8;
    value |= (vc_inb(FW_CFG_DATA) as u32) << 16;
    value |= (vc_inb(FW_CFG_DATA) as u32) << 24;

    value
}
```
这个函数同样采用了协议`GHCB_NAE_IOIO`，实现了数据的读取，并存在返回值。
```rust
pub fn vc_inb(port: u16) -> u8 {
    let ghcb: *mut Ghcb = vc_get_ghcb();
    let mut ioio: u64 = (port as u64) << 16;
    let value: u8;

    // 64位系统，8位操作数，In方式进行access
    ioio |= IOIO_ADDR_64;
    ioio |= IOIO_SIZE_8;
    ioio |= IOIO_TYPE_IN;

    unsafe {
        (*ghcb).set_rax(0);

        vc_perform_vmgexit(ghcb, GHCB_NAE_IOIO, ioio, 0);

        if !(*ghcb).is_rax_valid() {
            vc_terminate_svsm_resp_invalid();
        }
        // 返回值会存放在value之中
        value = LOWER_8BITS!((*ghcb).rax()) as u8;

        (*ghcb).clear();
    }

    value
}
```
通过多次拼凑，将`32`位的`value`装配出来。这个`value`在后续就会有用。

一般而言，似乎`OUT`与`IN`会进行配合使用。上面的验签就似乎是一个很好的例子。
#### perform_dma:
这个函数具体做了什么非常值得了解：
```rust
fn perform_dma(dma: &mut FwCfgDma, data: *const u8, control: u32, size: usize) {
    // 虚实地址转化，以实打实地把相关数据写进去
    let dma_pa = pgtable_va_to_pa(VirtAddr::new(dma as *mut FwCfgDma as u64));
    let dma_data_pa = pgtable_va_to_pa(VirtAddr::new(&dma.data as *const u8 as u64));

    assert!(size <= DMA_DATA_SIZE);

    // 为什么要调换一下顺序，可能和DMA的读取方式有关系
    // 按照外部设备的字节读取方式来读取数据，之后再存储在系统之中
    dma.desc.control = u32::swap_bytes(control);
    dma.desc.length = u32::swap_bytes(size as u32);
    dma.desc.address = u64::swap_bytes(dma_data_pa.as_u64());

    // 拆成两半，将dma_pa地址分贝输出到对应的FW_CFG_DMA_HI和FW_CFG_DMA_LO端口上
    let lo: u32 = LOWER_32BITS!(dma_pa.as_u64()) as u32;
    let hi: u32 = UPPER_32BITS!(dma_pa.as_u64()) as u32;
    vc_outl(FW_CFG_DMA_HI, u32::swap_bytes(hi));
    vc_outl(FW_CFG_DMA_LO, u32::swap_bytes(lo));
    // 因为DMA能够直接将数据读取到特定的位置，并不需要通知CPU去获取这些信息，因此与一般情况下交互的out/in流程不太一样
    // 因此在这边并不需要直接通过in函数来将数据读取出来，他们会自动地放在对应的物理地址相关位置
    let mut c: u32;
    loop {
        c = u32::swap_bytes(dma.desc.control);
        if (c & !FW_CFG_DMA_ERROR) == 0 {
            break;
        }
        pause();
    }

    if (c & FW_CFG_DMA_ERROR) != 0 {
        vc_terminate_svsm_fwcfg();
    }

    unsafe {
        // 再把数据写回到data之中
        let p: *mut u8 = data as *mut u8;
        copy_nonoverlapping(&dma.data as *const u8, p, size);
    }
}
```
至此，针对`fwcfg_init`函数相关的解读任务也完成了。