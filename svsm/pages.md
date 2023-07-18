## 有关于pagetable_init函数的分析
由于代码的阅读工作量似乎比我想象的大一些，姑且不再尝试一口气吃成一只大胖子了呜呜呜，我们尝试慢慢啃。可以尝试每天花四个小时读源码。
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
在这一则中，我们主要集中精力分析函数`pgtable_init()`，注释代号记为`pt`。这个函数展开后如下所示：
```rust
/// Generate 4-level page table, update Cr3 accordingly
pub fn pgtable_init() {
    lazy_static::initialize(&PGTABLE);

    let flags: PageTableFlags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
    let mut allocator: PageTableAllocator = PageTableAllocator::new();
    unsafe {
        __pgtable_init(flags, &mut allocator);
    }
}
```
这里我们简单向下分析几个点：
### PGTABLE是怎么来的：
```rust
lazy_static! {
    static ref PGTABLE: SpinLock<OffsetPageTable<'static>> = {
        unsafe {
            let pgt: OffsetPageTable = OffsetPageTable::new(&mut P4, SVSM_GVA_OFFSET);
            SpinLock::new(pgt)
        }
    };
}
```
围绕的函数核心是第一行`lazy_static`对于该数据结构的初始化。其中`spinlock`是为了让多线程环境下对共享资源实现互斥访问，`'static`关键字则是一种生命周期长达整个程序运行期间的全局变量，它们在程序启动的时候被初始化。

而这边的`OffsetPageTable`也是有一些讲究的。其中`PhysOffset`参数表示该页表的物理地址
```rust
/// A Mapper implementation that requires that the complete physically memory is mapped at some
/// offset in the virtual address space.
#[derive(Debug)]
pub struct OffsetPageTable<'a> {
    inner: MappedPageTable<'a, PhysOffset>,
}

/// A Mapper implementation that relies on a PhysAddr to VirtAddr conversion function.
///
/// This type requires that the all physical page table frames are mapped to some virtual
/// address. Normally, this is done by mapping the complete physical address space into
/// the virtual address space at some offset. Other mappings between physical and virtual
/// memory are possible too, as long as they can be calculated as an `PhysAddr` to
/// `VirtAddr` closure.
#[derive(Debug)]
pub struct MappedPageTable<'a, P: PageTableFrameMapping> {
    page_table_walker: PageTableWalker<P>,
    level_4_table: &'a mut PageTable,
}
```
`P`表示的是拥有`trait`类型为`PageTableFrameMapping`的数据结构，在这边引用的例子是`PhysOffset`，如下所示：
```rust
// 以VirtAddr进行表示的原因似乎是，恰好是48位有效。如果是PhysAddr，对应的是52位有效
#[derive(Debug)]
struct PhysOffset {
    offset: VirtAddr,
}

// 以PhysOffset为基准，去找到frame的起始地址并与之相加，进而找到映射的虚拟地址指针位置
unsafe impl PageTableFrameMapping for PhysOffset {
    fn frame_to_pointer(&self, frame: PhysFrame) -> *mut PageTable {
        let virt = self.offset + frame.start_address().as_u64();
        virt.as_mut_ptr()
    }
}
```
我们可以通过初始化的`PhysOffset`来锁定对应页表的位置，以及页表`frame`的位置。

`PageTableWalker`则是用于遍历页表的`cursor`类物件，其可以通过`next_table`和`create_next_table`的方式对页表实现遍历的操作，当然，要附上对应的`PageTabelEntry`，注意这里的`next_table`指的是下一级页表。

简单看一下`PageTable`是怎么做的，如下所示：
```rust
/// Represents a page table.
///
/// Always page-sized.
///
/// This struct implements the `Index` and `IndexMut` traits, so the entries can be accessed
/// through index operations. For example, `page_table[15]` returns the 15th page table entry.
///
/// Note that while this type implements [`Clone`], the users must be careful not to introduce
/// mutable aliasing by using the cloned page tables.
#[repr(align(4096))]
#[repr(C)]
#[derive(Clone)]
pub struct PageTable {
    entries: [PageTableEntry; ENTRY_COUNT],
}
```
或许这是最精简的一种表示多级页表的方式了，直接采用数组来做，不过没有被分配的页表项会以`EMPTY`类型示人，此外还有相关的函数帮助实现页表项的添加，清空，遍历等操作。`PageTableEntry`则是页表中页表项的数据类型，可以设置页表项映射到哪些地址上去，其中存放了一些`Flag`信息，可以自由读取出来。

当然，就`PageTableEntry`的相关操作函数来看，其存在的自由度很大，考虑到页表分配时的空间局部性问题，真实情况下应该是“带着镣铐跳舞”。

在`PageTableEntry`之中还有存在着一类信息名为`PageTableFlags`，它的数据结构是内嵌了`bitflags`宏的类型，这方面我们查看`Intel`的手册大致会有一个不错的印象。

有了上述的信息准备，我们可以看到`OffsetPageTable`的建立流程。此处的Offset其实就是`0xffff800000000000`，这样就和下面的数据结构联系起来了。下面的数据结构是一种特殊的页表，其物理地址和虚拟地址维持着以下的偏移量关系。
```rust
impl<'a> OffsetPageTable<'a> {
    /// Creates a new `OffsetPageTable` that uses the given offset for converting virtual
    /// to physical addresses.
    ///
    /// The complete physical memory must be mapped in the virtual address space starting at
    /// address `phys_offset`. This means that for example physical address `0x5000` can be
    /// accessed through virtual address `phys_offset + 0x5000`. This mapping is required because
    /// the mapper needs to access page tables, which are not mapped into the virtual address
    /// space by default.
    ///
    /// ## Safety
    ///
    /// This function is unsafe because the caller must guarantee that the passed `phys_offset`
    /// is correct. Also, the passed `level_4_table` must point to the level 4 page table
    /// of a valid page table hierarchy. Otherwise this function might break memory safety, e.g.
    /// by writing to an illegal memory location.
    #[inline]
    pub unsafe fn new(level_4_table: &'a mut PageTable, phys_offset: VirtAddr) -> Self {
        let phys_offset = PhysOffset {
            offset: phys_offset,
        };
        Self {
            inner: unsafe { MappedPageTable::new(level_4_table, phys_offset) },
        }
    }

    /// Returns a mutable reference to the wrapped level 4 `PageTable` instance.
    pub fn level_4_table(&mut self) -> &mut PageTable {
        self.inner.level_4_table()
    }

    /// Returns the offset used for converting virtual to physical addresses.
    pub fn phys_offset(&self) -> VirtAddr {
        self.inner.page_table_frame_mapping().offset
    }
}
```
之后，再用`spinlock`对刚刚生成的`offsetpagetable`进行封装就可以了。
### PageTableAllocator的建立：
这一数据类型似乎被分配了一些其余特性，我们可以看到如下的情况：
```rust
#[derive(Copy, Clone, Debug)]
struct PageTableAllocator {}

impl PageTableAllocator {
    pub const fn new() -> Self {
        PageTableAllocator {}
    }
}

unsafe impl FrameAllocator<Size4KiB> for PageTableAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        mem_allocate_frame()
    }
}

impl FrameDeallocator<Size4KiB> for PageTableAllocator {
    unsafe fn deallocate_frame(&mut self, frame: PhysFrame) {
        mem_free_frame(frame)
    }
}
```
就建立来说似乎不太困难，不过似乎有必要去了解一下底层的内存是怎么分配的，检查代码我们感觉似乎和`SlabPage`有一定的关联。

这边的流程就和很多小项目，如`WeensyOS`的`kalloc`一样，我们可以看到下面的代码情况：
```rust
pub fn mem_allocate_frame() -> Option<PhysFrame> {
    mem_allocate_frames(1)
}

pub fn mem_allocate_frames(count: u64) -> Option<PhysFrame> {
    let order: usize = SvsmAllocator::get_order((count * PAGE_SIZE) as usize);
    let result = ROOT_MEM.lock().allocate_pages(order);

    let frame = match result {
        Ok(vaddr) => Some(PhysFrame::from_start_address(pgtable_va_to_pa(vaddr)).unwrap()),
        Err(_e) => None,
    };

    if let Some(f) = frame {
        let vaddr: VirtAddr = pgtable_pa_to_va(f.start_address());
        unsafe {
            let dst: *mut u8 = vaddr.as_mut_ptr();
            core::intrinsics::write_bytes(dst, 0, (PAGE_SIZE << order) as usize);
        }
    }

    frame
}
```
借助了`SvsmAllocator`进行内存分配，根据内存大小在`get_order`中获取`order`值，之后利用`order`在`ROOT_MEM`之中分配对应的内存空间。
```rust
pub fn mem_allocate_frames(count: u64) -> Option<PhysFrame> {
    let order: usize = SvsmAllocator::get_order((count * PAGE_SIZE) as usize);
    let result = ROOT_MEM.lock().allocate_pages(order);

    let frame = match result {
        Ok(vaddr) => Some(PhysFrame::from_start_address(pgtable_va_to_pa(vaddr)).unwrap()),
        Err(_e) => None,
    };

    if let Some(f) = frame {
        let vaddr: VirtAddr = pgtable_pa_to_va(f.start_address());
        unsafe {
            let dst: *mut u8 = vaddr.as_mut_ptr();
            core::intrinsics::write_bytes(dst, 0, (PAGE_SIZE << order) as usize);
        }
    }

    frame
}
```
对于`ROOT_MEM`类型，我们不再分析，之前看过相关的源码：此处相关的函数为`allocate_pages`，可以一看：
```rust
static ROOT_MEM: SpinLock<MemoryRegion> = SpinLock::new(MemoryRegion::new());

    pub fn allocate_pages(&mut self, order: usize) -> Result<VirtAddr, ()> {
        if order >= MAX_ORDER {
            return Err(());
        }
        self.refill_page_list(order)?;
        if let Ok(pfn) = self.get_next_page(order) {
            let pg: SvsmPageInfo = SvsmPageInfo::Allocated(AllocatedInfo { order: order });
            self.write_page_info(pfn, pg);
            let vaddr: VirtAddr = self.start_virt + (pfn * PAGE_SIZE as usize);
            return Ok(vaddr);
        } else {
            return Err(());
        }
    }
```
完犊子，地动山摇了，这里还是先做一下知识屏蔽吧。总之这边利用`slab`化的内存空间进行分配，然后得到`frames`并把里子清空了，就完事儿了。至于`deallocte_pages`，说白了他是从事物的反面来做的，或许我们可以不再赘述了。

就看一下代码好了：
```rust
pub fn mem_free_frames(frame: PhysFrame, _count: u64) {
    let vaddr: VirtAddr = pgtable_pa_to_va(frame.start_address());
    free_page(vaddr);
}

pub fn mem_free_frame(frame: PhysFrame) {
    mem_free_frames(frame, 1);
}

fn free_page(vaddr: VirtAddr) {
    ROOT_MEM.lock().free_page(vaddr)
}

    pub fn free_page(&mut self, vaddr: VirtAddr) {
        let res = self.get_page_info(vaddr);

        if let Err(_e) = res {
            return;
        }

        let pfn: usize = ((vaddr - self.start_virt) / PAGE_SIZE) as usize;

        match res.unwrap() {
            SvsmPageInfo::Allocated(ai) => {
                self.free_page_order(pfn, ai.order);
            }
            SvsmPageInfo::SlabPage(_si) => {
                self.free_page_order(pfn, 0);
            }
            _ => {
                panic!("Unexpected page type in MemoryRegion::free_page()");
            }
        }
    }
```
反正最终追溯下来一定是和`Slab`分配器有点关系的，这玩意儿我只在`IPADS`的书上看过，具体代码确实没有看过，有空可以瞅一瞅。
### 页表初始化：
有了这些准备，我们终于可以尝试去分析整个页表被初始化流程了。首先需要注意，`Frame`用于物理地址，而`Page`用于虚拟地址，他们都表示一个地址块类似物。

这里可能有一小部分会容易感觉到比较奇怪，特别的，`sev_encryption_mask`并不全部都是`47`位，这一个信息我们需要利用CPUID来获得，可以参考https://www.amd.com/system/files/TechDocs/24594.pdf 即`volume 3`的第`678`页。对于看代码而言，我们可以暂且忽略这个信息，只要知道映射到了`private pa`，即机密物理内存`frame`上就可以了。
```rust
// __pgtable_init
unsafe fn __pgtable_init(flags: PageTableFlags, allocator: &mut PageTableAllocator) {
    let mut va: VirtAddr = get_svsm_begin();
    let va_end: VirtAddr = get_svsm_end();
    // 针对全部的virtAddr进行映射的工作
    while va < va_end {
        let pa: PhysAddr = pgtable_va_to_pa(va);
        let private_pa: PhysAddr = PhysAddr::new(pa.as_u64() | get_sev_encryption_mask());
        // 从虚拟地址，以及将要映射到的物理地址，计算得到他们的page和frame号，以方便后续的映射
        let page: Page<Size4KiB> = page_with_addr(va);
        let frame: PhysFrame = PhysFrame::from_start_address_unchecked(private_pa);

        let result: Result<MapperFlush<Size4KiB>, MapToError<Size4KiB>> = PGTABLE
            .lock()
            .map_to_with_table_flags(page, frame, flags, flags, allocator);
        if result.is_err() {
            vc_terminate_ghcb_general();
        }

        va += PAGE_SIZE;
    }

    // Change the early GHCB to shared for use before a new one is created
    // 一开始的GHCB被设置成了RESCINDE类型
    let va: VirtAddr = get_early_ghcb();
    let page: Page<Size4KiB> = page_with_addr(va);
    remap_page(page, PageType::Shared, false);

    // Mark the BSS and DATA sections as non-executable
    // 将数据段和BSS段全部设置为无法运行的状态
    let mut set: PageTableFlags = PageTableFlags::NO_EXECUTE;
    let mut clr: PageTableFlags = PageTableFlags::empty();
    // bss段
    let mut page_begin: Page<Size4KiB> = page_with_addr(get_svsm_sbss());
    let mut page_end: Page<Size4KiB> = page_with_addr(get_svsm_ebss());
    update_page_flags(Page::range(page_begin, page_end), set, clr, false);
    // data段
    page_begin = page_with_addr(get_svsm_sdata());
    page_end = page_with_addr(get_svsm_edata());
    update_page_flags(Page::range(page_begin, page_end), set, clr, false);
    // dyn_mem段
    // 总之我们要求能跑的也就code段！
    page_begin = page_with_addr(get_dyn_mem_begin());
    page_end = page_with_addr(get_dyn_mem_end());
    update_page_flags(Page::range(page_begin, page_end), set, clr, false);

    // Mark the BSP stack guard page as non-present
    set = PageTableFlags::empty();
    clr = PageTableFlags::PRESENT;
    // guard page大小正好是4096 字节 这么大
    page_begin = page_with_addr(get_guard_page());
    page_end = page_begin + 1;
    update_page_flags(Page::range(page_begin, page_end), set, clr, false);

    // Use the new page table
    let p4: VirtAddr = VirtAddr::new(&P4 as *const PageTable as u64);
    let p4_pa: PhysAddr = PhysAddr::new(pgtable_va_to_pa(p4).as_u64() | get_sev_encryption_mask());
    let cr3: PhysFrame = PhysFrame::containing_address(p4_pa);
    // cr3 自然会被写为page table起始位置
    Cr3::write(cr3, Cr3Flags::empty());
}
```
到这里，我们对于页表的代码解读便结束了。
