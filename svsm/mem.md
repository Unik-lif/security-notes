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
设置`svsm`的内存`pages`们为`private`状态

在这边我们需要研究一下函数`vc_early_make_pages_private`，如下所示：这个函数从`svsm`动态内存的`PhysFrame`起始位置开始，一直遍历到`PhysFrame`的结束位置。

内存以`Frame`进行分配。
```rust
// 该函数把Ghcb记作一个可变项
pub fn vc_early_make_pages_private(begin: PhysFrame, end: PhysFrame) {、
    // 搞到early_ghcb中的ghcb地址，准备修改它
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
        // 实际上，根据build_psc_entries的函数结构来观察，这个循环感觉只会进行一次
        pa = build_psc_entries(&mut op, pa, pa_end, page_op);

        let last_entry: u16 = op.header.end_entry;

        // 如果传递的参数page_op是PSC_SHARED类型，则采用下面的方式，根据op存放的物理页信息，对相关的物理页进行pvalidate操作
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
            // size对应的部分其实是管理者PscOp的全部信息内容，说白了就是一个像管理向量一样的东西
            // 通过函数set_shared_buffer把长度为size的set_bytes复制到ghcb的shared_buffer之中
            // 将op内用于管理的那些数据整体赋值到ghcb之中
            // 之后，把ghcb的sw_scratch设置为shared_buffer对应的虚拟地址
            (*ghcb).set_shared_buffer(set_bytes, size);

            while op.header.cur_entry <= last_entry {
                // 设置GHCB_NAE_PSC并退出，调用vmmcall指令
                // psc是svsm之上的组件
                // 对GHCB相关的组件做简单的设置
                // vmgexit还会通过vmmcall通知hypervisor得知相关的情况
                vc_perform_vmgexit(ghcb, GHCB_NAE_PSC, 0, 0);
                if !(*ghcb).is_sw_exit_info_2_valid() || (*ghcb).sw_exit_info_2() != 0 {
                    vc_terminate_svsm_psc();
                }
                // 继续执行拷贝操作，东西记录在ghcb之中
                (*ghcb).shared_buffer(get_bytes, size);
            }
        }

        // 如一开始就打算作为私有，则很快就能把它设置为VALIDATE状态
        // 记作shared的GHCB本身应该被设置为RESCINE状态
        // 为什么这么设置确实有待继续对源码进行解读
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

在有了这个存放信息用的结构体后，我们用`build_psc_entries`函数进行来对`svsm`动态内存的全部物理页进行检视。在`svsm`的建立过程中，优先采用`2MB`的大页形式来进行存储。
```rust
// 可以把op当做一个管理者
// 在build_psc_entries内部实现了逐个页面遍历并且赋上page_op属性的操作
// 直到遍历完成，此时在op.entries[i]内将会存放各个页对应的信息，2MB和4KB不同的页面将会以不同的方式来进行存储
// 他们的区别可以通过GHCB_2MB等相关宏进行区分，相关数据将会存放在data段之中，这边做的方式很直接，直接 | 上一个2 ^ 56，从而彻底避开了干扰
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
        // 随时更新头文件中的end_entry序号
        op.header.end_entry = i as u16;

        i += 1;
    }

    return pa;
}
```
在上面的函数中提及`pvalidate_psc_entries`，目测是通过`pvalidate`方式处理相关的物理内存，并赋上相关的权限。
```rust
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
        // 如果大小设置上size和page对应的size并没有匹配上，将会返回PVALIDATE_FAIL_SIZE_MISMATCH的报错
        // 这边处理错误的流程正好对应得上15-40的表格，RMP page size为4KB，而错误地以2MB的方式去处理它了
        if ret == PVALIDATE_FAIL_SIZE_MISMATCH && size > 0 {
            let va_end = va + PAGE_2MB_SIZE;
            // size项设置为0，表示用逐个的4KB页来处理
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
### root_mem_init函数
将`svsm`的`dyn_mem`相关的信息以`op`的形式在`early_ghcb`所对应的情况设置完备后，我们现在看一下函数`root_mem_init`：
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
`ROOT_MEM`是一块标准的地址区域，其包含信息如下：大体上遵循`buddy_system_allocator`的方式进行内存上的分配。
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

static ROOT_MEM: SpinLock<MemoryRegion> = SpinLock::new(MemoryRegion::new());
```
在完成了上述的初始化工作后，我们进入函数`init_memory()`，如下所示：
```rust
pub fn init_memory(&mut self) {
    // buddy_system对于每个大小为PAGE_SIZE的页均采用八个字节进行存储
    // page_count在ROOT_MEM被初始化时就已经被写入了
    // size即为用于存放全部正常工作的内存PAGE信息以metadata方式存储时的总大小
    let item_size: usize = size_of::<PageStorageType>();
    let size: u64 = (self.page_count * item_size) as u64;
    let meta_pages: usize = (align_up(size, PAGE_SIZE) / PAGE_SIZE) as usize;

    /* Mark page storage as reserved */
    // meta pages用于存放页相关信息，额外算是一个存储空间
    // 需要把他们的PageInfo状态记作Reserved，用以与正常的页进行区分
    for i in 0..meta_pages {
        let pg: SvsmPageInfo = SvsmPageInfo::Reserved(ReservedInfo {});
        self.write_page_info(i, pg);
    }

    // buddy system中的order 0对应的是4 KB的数目，这恰好是最小的内存管理单元
    self.nr_pages[0] = self.page_count - meta_pages;

    /* Mark all pages as allocated */
    // 其余的部分被设置为被分配的状态，order表示的是buddy system中的粒度
    // 当然，全部都在访问metadata_page部分的内容
    for i in meta_pages..self.page_count {
        let pg: SvsmPageInfo = SvsmPageInfo::Allocated(AllocatedInfo { order: 0 });
        self.write_page_info(i, pg);
    }

    /* Now free all pages */
    // 一开始buddy_system还没有对内存进行合并时，均存放在order 0之中
    // 接下来需要通过free_page_order来尝试对相关的内存进行合并
    // 这是buddy_system中的用法
    // 注意：在开始的时候，这些SVSMPageInfo状态都被设置成了Allocated
    for i in meta_pages..self.page_count {
        self.free_page_order(i, 0);
    }
}
```
关键函数`write_page_info`如下所示：
```rust
fn write_page_info(&self, pfn: usize, pi: SvsmPageInfo) {
    self.check_pfn(pfn);

    // to_mem将会把svsm page info转变为page storage type类型信息
    let info: PageStorageType = pi.to_mem();
    unsafe {
        // 先进行地址转换，当然去找的是metadata_page内的用于存放类型的位置
        let ptr: *mut PageStorageType =
            self.page_info_virt_addr(pfn).as_u64() as *mut PageStorageType;
        // 再进行信息存储
        (*ptr) = info;
    }
}

// 检查pfn数是否超过了page_count
fn check_pfn(&self, pfn: usize) {
    if pfn >= self.page_count {
        panic!("Invalid Page Number {}", pfn);
    }
}

// 根据pfn，找到对应的virt地址
// 映射关系如下所示，start_virt与start_phys是直接对应着的，而start_phys与page_info之间的偏移量为pfn * size
fn page_info_virt_addr(&self, pfn: usize) -> VirtAddr {
    let size: usize = size_of::<PageStorageType>();
    let virt: VirtAddr = self.start_virt;
    virt + ((pfn as usize) * size)
}
```
函数`free_page_order`如下所示：
```rust
fn free_page_order(&mut self, pfn: usize, order: usize) {
    match self.try_to_merge_page(pfn, order) {
        Err(_e) => {
            self.free_page_raw(pfn, order);
        }
        Ok(new_pfn) => {
            self.free_page_order(new_pfn, order + 1);
        }
    }
}

// 尝试与buddy进行一次merge操作，以获得更大的空闲内存块
fn try_to_merge_page(&mut self, pfn: usize, order: usize) -> Result<usize, ()> {
    // 找到邻居的pfn号，确定相关的内存
    let neighbor_pfn: usize = self.compound_neighbor(pfn, order)?;
    let neighbor_page: SvsmPageInfo = self.read_page_info(neighbor_pfn);

    // 仅在svsmpageinfo为free的情况下，才能进行合并
    // 初始化时候的页面状态都为allocated，因此没法合并，会自动返回err
    // 我们暂时不再分析allocate_pfn函数，在有必要的时候回来分析allocate_pfn和merge_pages函数即可
    if let SvsmPageInfo::Free(fi) = neighbor_page {
        if fi.order != order {
            return Err(());
        }

        self.allocate_pfn(neighbor_pfn, order)?;

        let new_pfn: usize = self.merge_pages(pfn, neighbor_pfn, order)?;

        Ok(new_pfn)
    } else {
        Err(())
    }
}

// 找到邻居的位置
fn compound_neighbor(&self, pfn: usize, order: usize) -> Result<usize, ()> {
    if order >= MAX_ORDER - 1 {
        return Err(());
    }

    // 确定vaddr,pfn_to_virt本质上也只是从start_virt相加一个偏移量
    // order_mask则是帮助直接以order pages的方式来锁定一块内存块的起始位置
    let vaddr: VirtAddr =
        VirtAddr::new(self.pfn_to_virt(pfn).as_u64() & MemoryRegion::order_mask(order));
    // 邻居的虚拟地址则是直接在order位上的PAGE_SIZE做一次简单的翻转即可，二者紧密相连
    let neigh: VirtAddr = VirtAddr::new(vaddr.as_u64() ^ (PAGE_SIZE << order));
    // 确定vaddr和neigh的地址是否合法
    if vaddr < self.start_virt || neigh < self.start_virt {
        return Err(());
    }
    // 确定具体neigh的pfn序号
    let pfn: usize = self.virt_to_pfn(neigh);
    if pfn >= self.page_count {
        return Err(());
    }

    Ok(pfn)
}

// read_page_info函数
// 读取svsmpageinfo状态
fn read_page_info(&self, pfn: usize) -> SvsmPageInfo {
    self.check_pfn(pfn);

    let virt: VirtAddr = self.page_info_virt_addr(pfn);
    let info: PageStorageType = PageStorageType(unsafe { *(virt.as_u64() as *const u64) });

    SvsmPageInfo::from_mem(info)
}

// allocate_pfn，利用buddy_system进行内存分配的步骤
fn allocate_pfn(&mut self, pfn: usize, order: usize) -> Result<(), ()> {
    // 首先检查下一个用于分配的next_page之中，是否有order级别大小的页存在
    let first_pfn: usize = self.next_page[order];

    // Handle special cases first
    if first_pfn == 0 {
        // No pages for that order
        return Err(());
    } else if first_pfn == pfn {
        // Requested pfn is first in list
        self.get_next_page(order).unwrap();
        return Ok(());
    }

    // Now walk the list
    let mut old_pfn: usize = first_pfn;
    loop {
        let current_pfn: usize = self.next_free_pfn(old_pfn, order);
        if current_pfn == 0 {
            break;
        } else if current_pfn == pfn {
            let next_pfn: usize = self.next_free_pfn(current_pfn, order);
            let pg: SvsmPageInfo = SvsmPageInfo::Free(FreeInfo {
                next_page: next_pfn,
                order: order,
            });
            self.write_page_info(old_pfn, pg);

            let pg: SvsmPageInfo = SvsmPageInfo::Allocated(AllocatedInfo { order: order });
            self.write_page_info(current_pfn, pg);

            self.free_pages[order] -= 1;

            return Ok(());
        }

        old_pfn = current_pfn;
    }

    return Err(());
}

// 初始化的时候会进入这个函数之中
fn free_page_raw(&mut self, pfn: usize, order: usize) {
    let old_next: usize = self.next_page[order];
    let pg: SvsmPageInfo = SvsmPageInfo::Free(FreeInfo {
        next_page: old_next,
        order: order,
    });

    self.write_page_info(pfn, pg);
    self.next_page[order] = pfn;
    // 这样一来free_pages的数目将会得到更新
    self.free_pages[order] += 1;
}
```
在完成了内存初始化的同时，函数也对`slab`内存分配器做了一些初始化的工作。简单看一下`SLAB_PAGE_SLAB`的初始化流程。
```rust
static SLAB_PAGE_SLAB: SpinLock<Slab> = SpinLock::new(Slab::new(size_of::<SlabPage>() as u16));

impl Slab {    
    // 针对Slab的初始化的函数如下所示
    pub const fn new(item_size: u16) -> Self {
        Slab {
            item_size: item_size,
            capacity: 0,
            free: 0,
            pages: 0,
            full_pages: 0,
            free_pages: 0,
            page: SlabPage::new(),
        }
    }
    
    pub fn init(&mut self) -> Result<(), ()> {
        let slab_vaddr: VirtAddr = VirtAddr::new((self as *mut Slab) as u64);
        if let Err(_e) = self.page.init(slab_vaddr, self.item_size) {
            return Err(());
        }

        self.capacity = self.page.get_capacity() as u32;
        self.free = self.capacity;
        self.pages = 1;
        self.full_pages = 0;
        self.free_pages = 1;

        Ok(())
    }
}
```
这个东西比较复杂，我们就不再深究了。

总而言之，在`mem_init`函数中，我们通过读`ghcb`块，并利用`pvalidate`命令将其设置为用户所私有的块，并对`root_mem`做了一些初始化工作，也初始化了`slab allocator`。

这里的代码细节设计到一部分关于`slab`和`buddy_system`的知识，它们掌握起来并没有那么容易，我们姑且在有空的时候再通过文献阅读和撰写代码来更清楚地认识这件事情。