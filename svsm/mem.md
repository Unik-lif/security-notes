## 有关于mem_init函数的笔记整理
这个函数大体上如下所示：首先检测此处的`MAX_ORDER`和`REAL_MAX_ORDER`大小是否合理，即`MAX_ORDER`是否超标。
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
```
根据注释，我们接下来会用`__mem_init`来实现内存分配，这个结构应该是一个`buddy_system_allocator`结构。
```rust
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
### vc_early_make_pages_private函数
在这边我们需要研究一下函数`vc_early_make_pages_private`，如下所示：这个函数从`svsm`动态内存的`PhysFrame`起始位置开始，一直遍历到`PhysFrame`的结束位置。

内存以`Frame`进行分配。
```rust
// 该函数把Ghcb记作一个可变项
pub fn vc_early_make_pages_private(begin: PhysFrame, end: PhysFrame) {、
    // 搞到early_ghcb地址，准备修改它
    let ghcb: *mut Ghcb = get_early_ghcb().as_mut_ptr() as *mut Ghcb;

    // 尝试把begin和end区域内的信息拷贝到ghcb中
    perform_page_state_change(ghcb, begin, end, PSC_PRIVATE);
}
```
函数`perform_page_state_change`用于将`begin`和`end`区域内的物理内存拷贝到`ghcb`之中。
```rust
fn perform_page_state_change(ghcb: *mut Ghcb, begin: PhysFrame, end: PhysFrame, page_op: u64) {
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
    while pa < pa_end {
        // 设置op的头为cur_entry，表示从0开始向下遍历
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
```
在这边函数中创造的`PscOp`值得我们研究，该数据结构如下所示：
```rust
#[allow(dead_code)]
impl PscOp {
    pub const fn new() -> Self {
        let h: PscOpHeader = PscOpHeader::new();
        let d: PscOpData = PscOpData::new();

        PscOp {
            header: h,
            entries: [d; PSC_ENTRIES],
        }
    }
    funcs!(header, PscOpHeader);
    funcs!(entries, [PscOpData; PSC_ENTRIES]);
}
```
`PscOp`由`header`和`entries`共同组成，对于`Header`和`Data`，其数据结构如下所示：
```rust
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct PscOpHeader {
    pub cur_entry: u16,
    pub end_entry: u16,
    pub reserved: u32,
}

#[allow(dead_code)]
impl PscOpHeader {
    pub const fn new() -> Self {
        PscOpHeader {
            cur_entry: 0,
            end_entry: 0,
            reserved: 0,
        }
    }
    funcs!(cur_entry, u16);
    funcs!(end_entry, u16);
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct PscOpData {
    pub data: u64,
}

#[allow(dead_code)]
impl PscOpData {
    pub const fn new() -> Self {
        PscOpData { data: 0 }
    }
    funcs!(data, u64);
}
```
`PscOp`像是一个用于存放信息的数组，其中`header`中包含了`cur_entry`和`end_entry`等相关信息，以方便在`Data`的管理数组`Entries`之中进行访问与遍历。

在有了这个存放信息用的结构体后，我们用`build_psc_entries`函数进行来对`svsm`动态内存的全部物理页进行检视。在`svsm`的建立过程中，优先采用`2MB`的大页形式
```rust
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