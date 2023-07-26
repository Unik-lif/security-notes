## start.S
这个是最初的启动代码，所涉及细节比较多，我们直接以注释的形式进行解读。
```assembly
/* SPDX-License-Identifier: MIT */
/*
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
 *
 * Author: Tom Lendacky <thomas.lendacky@amd.com>
 */

// 调用宏svsm.h之中的信息
#include "svsm.h"

// 这个原来是C-bit哈哈哈，搞明白了
// 通过设置C-bit来实现加密，物理地址会被加上C-bit，表示启用了内存加密
// 在加密开启之后，真实数据的存放位置并不会发生改变，但是内存页上的地址会变化
#define CBIT(x)			(BIT(51) + x)
#define GPA(x)			(x - SVSM_GVA_OFFSET_ASM)

// 可以理解成SVSM要求采用特殊的某些页表来完成映射任务
// 搞不清楚的问题：为啥会有0x03，0x83这样的设置，可能是一些特殊的设置
#define SVSM_PGD_ENTRY(x)	(CBIT(x) + 0x03)
#define SVSM_P4D_ENTRY(x)	(CBIT(x) + 0x03)
#define SVSM_PUD_ENTRY(x)	(CBIT(x) + 0x03)
#define SVSM_PMD_ENTRY(x)	(CBIT(x) + 0x83)
#define SVSM_PTE_ENTRY(x)	(CBIT(x) + 0x03)

	.code64
GLOBAL(code_64)
	cli // close the interrupts.
	// 让我所感到比较奇怪的事情是，这页表到底是什么时候做出来的？一般来说启动的时候还需要动一动标志位之类的。

	xorq	%rax, %rax
	movq	%rax, %ds
	movq	%rax, %es
	movq	%rax, %fs
	movq	%rax, %ss

	/* Setup a stack */
	movq	cpu_stack(%rip), %rsp

/*
 * Jump to main high-level language code now for APs
 * --------------------------------------------------
 * cpu_mode is set to 0 before, so we will skip this.
 */
	cmpl	$0, cpu_mode(%rip)
	jne	hl

	/* 
	 * Load RBX with the virtual address offset for use throughout boot 
	 * ----------------------------------------------------------------
	 * SVSM_GVA_OFFSET_ASM: 最高的47位地址空间，并且最高位设置成了1
	 * rbx存放了SVSM_GVA_OFFSET_ASM的数值
	 */
	movq	$SVSM_GVA_OFFSET_ASM, %rbx

	/* 
	 * GS is set for APs, only clear it after the AP check 
	 * ---------------------------------------------------
	 * gs寄存器用于TLS，即作为线程存放自己本地存储空间的基地址
	 * 后面的代码之中我们确实看到GS和AP核息息相关
	 */
	movq	%rax, %gs

	/*
	 * SEV mitigation test to verify encryption bit position:
	 *   Use the CMP instruction, with RIP-relative addressing, to compare
	 *   the first four bytes of the CMP instruction itself (which will be
	 *   read decrypted if the encryption bit is in the proper location)
	 *   against the immediate value within the instruction itself
	 *   (instruction fetches are always decrypted by hardware).
	 * -------------------------------------------------------------
	 * 虽然这个解释感觉有点奇怪：
	 * 现在的问题是，需要验证加密位位置是否正确，如果不正确的话没法得到预期的正确解密后的值。
	 * 为此，我们比较insn位置读出来的指令信息，与0xfff63d81进行比较
	 */
	movq	$0x1f100, %rsi
insn:
	cmpl	$0xfff63d81, insn(%rip)
	jne	terminate_64

	/* 
	 * Validate that the build load address matches the actual load address 
	 * -------------------------------------------------------------
	 * 将rax的虚拟地址和rbx进行相减，rbx对应了SVSM_GVA起始位置，所以这边应该是检查
	 * rax被装载到的位置是否在512 GB处，我们检查一下svsm.lds.S发现装载地址确乎如此。
	 * 但是装载地址并不等同于最终的制造地址，一个很好的例子便是JOS中的一些细节，所以在这边可以简单
	 * 判别一下。
	 */
	// 对于rsi的值的设置至今还是一个谜
	movq	$0x2f100, %rsi
	leaq	code_64(%rip), %rax
	// rax计算得到从SVSM_GVA_OFFSET起始位置到code_64之间的距离
	// 如果链接文件正常运作了，最终的SVSM_GPA_ASM和rax值会是相同的
	subq	%rbx, %rax
	movq	$SVSM_GPA_ASM, %rcx
	cmpq	%rax, %rcx
	jne	terminate_64

	/*
	 * Make the early GHCB shared:
	 *   - Since there is only one PGD/P4D/PUD entry, operate on just
	 *     the PMD entry that holds the early GHCB.
	 * ---------------------------------------------------------------
	 * 把ghcb的地址塞到了early_ghcb的位置处
	 * ghcb是一块地址区域，大小为4 KB
	 * early_ghcb是一个全局变量，存放了ghcb的地址
	 * 注释中提到的到底是内存空间到底多少大确实不知道，可能之后才能确定具体的页表映射情况
	 */
	leaq	ghcb(%rip), %rsi
	movq	%rsi, early_ghcb(%rip)

	/*
	 * Rescind the page validation from LAUNCH.
	 * -----------------------------------------
	 * 在本次启动过程中，我们将启动时所用的地址空间设置为RESCIND状态
	 * 查阅https://www.amd.com/system/files/TechDocs/24593.pdf：
	 * rcx: 4 KB rdx: RESCIND
	 */
	movq	%rsi, %rax
	movq	$0, %rcx
	movq	$0, %rdx
	.byte 0xf2,0x0f,0x01,0xff	/* pvalidate */
	jc	terminate_64

	/*
	 * Issue the Page State Change to make shared in the RMP.
	 * --------------------------------------------------------
	 * 利用rcx来作为参数输入者，读取msr寄存器后，会把结果返回在rax和rdx之中
	 * 之后利用了pushq将他们塞入到栈中
	 * 0xc001_0130表示开启GHCB协议功能
	 * 为什么这么做？因为要保住原值，然后我们再操作GHCB protocol，操作完恢复。
	 * 在刚刚，我们利用了pvalidate指令实现了对于GHCB的共享，现在用户对于GHCB所对应的数据块也有了访问的权限
	 * 在下面我们会根据GHCB的协议实现交互
	 * 此处参考的资料是：https://www.amd.com/system/files/TechDocs/56421-guest-hypervisor-communication-block-standardization.pdf
	 * 我们摘自原文：
	 * The guest can read and write the GHCB value through MSR 0xc001_0130. 
	 * The hypervisor must not intercept access to MSR 0xc001_0130; otherwise, the 
	 * guest will not be able to successfully establish the GHCB.
	 * 当然，类似的操作在rust代码中出现了很多次
	 */
psc:
	movq	$0xc0010130, %rcx
	rdmsr
	pushq	%rax
	pushq	%rdx

	/*
	 * 将rsi置为SECTIONS段到ghcb的偏移量，即early_ghcb指向的ghcb页所处在的物理地址
	 * rax设置为2 ^ 53 + 0x14
	 * rax加上上述偏移量，交给rdx
	 * rdx 右移32位，即我们仅挑rdx的上半部分
	 * 初步计算可以得到rdx肯定是2 MB，因为偏移量是.text段，并没有超过4 GB
	 * 此外rax最底下是0x14，其他位不知道
	 * 0x14被设置时，表示为SNP Page State需要发生改变的一类请求
	 * edx:eax：因此2 ^ 53表示GHCBData中对于Page设置成了shared类型，而GHCBData的51到12位被设置成Guest Physical Frame Number
	 * 这个Guest Physical Frame Number由两个寄存器拼凑而成，通过rsi进行传递
	 */
	subq	%rbx, %rsi
	movq	$2, %rax
	shlq	$52, %rax
	addq	$0x14, %rax
	addq	%rsi, %rax
	movq	%rax, %rdx
	shrq	$32, %rdx
	/*
	 * 这里修改了msr的值，通过重新传递rax和rdx值的方法来实现。
	 * 此外下面对rax和rdx的值的东西做了一次判别。
	 * 在以0x14进行写入后，理应需要用0x15作为应答，如果rdx不为0，则说明发生了错误
	 * rep; vmmcall 是一个特殊的调用方法vmgexit的字节码拼凑方法
	 * rep 并没有什么语义，其只是一块拼图
	 */
	wrmsr
	rep; vmmcall
	// vmmcall 会完成一次交互，返回结果会放在rdx与rax之中
	rdmsr
	cmpq	$0x15, %rax
	jne	terminate_64
	cmpq	$0, %rdx
	jne	terminate_64

	/* 
	 * 现在约等于是恢复了原本validate的状态
	 * 所以我对上面这段代码的看法是，本质上是pvalidate以后简单利用vmcall，将GHCB对应的空间设置为shared Page状态，并检查这一过程是否发生了异常
	 * 这一被共享的空间将会在后续的启动流程中被多次用到
	 */
	popq	%rdx
	popq	%rax
	wrmsr

	/*
	 * Build the PTE entries. Use the address of the early GHCB to
	 * obtain the start and end of the 2MB page in which it lives.
	 * --------------------------------------------------------------
	 * 在数据段写pte，自行利用ghcb来手动建立PTE entries，此处存在for loop结构
	 */
	leaq	ghcb(%rip), %rsi
	subq	%rbx, %rsi // ghcb物理地址
	andq	$PAGE_2MB_MASK, %rsi // 先获得底部的2 MB内的具体地址位置
	addq	$PAGE_2MB_SIZE, %rsi // 再往上找2 MB空间
	movq	$SVSM_PTE_ENTRY(0), %rax // SVSM_PTE_ENTRY: CBIT(0) + 0x03.
	addq	%rax, %rsi // rsi的最终目的是把ghcb物理地址塞到页表的最后一级

	leaq	pte(%rip), %rax // pte地址存放到rax之中
	addq	$PAGE_SIZE, %rax // pte找到其最大项，然后逐项减下来

	// 一共有512项，反正就是遍历一遍喽
	// 完成时在PTE中的每一项，8 Byte，
    // 会被附上rsi对应的物理地址，然后在512个pte项中做填充
	movq	$PAGE_TABLE_ENTRY_COUNT, %rcx
set_pte:
	subq	$PAGE_SIZE, %rsi
	subq	$PAGE_TABLE_ENTRY_SIZE, %rax
	movq	%rsi, (%rax)
	loop	set_pte

	/* Make GHCB page shared */
	// 此处的ghcb还是虚拟地址
	leaq	ghcb(%rip), %rsi
	movq	%rsi, %rax
	shrq	$PAGE_SHIFT, %rax
	// 确定PFN，不过是最低的512个数内的，因为虚拟地址和物理地址只有顶部不一样，所以确实也就无所谓
	// 反正and完了都长一样的
	andq	$PAGE_TABLE_INDEX_MASK, %rax
	// shift left
	shlq	$3, %rax
	leaq	pte(%rip), %rcx
	// rcx与rax相加，因为每个项的大小都是8个字节，以此找到PTE内对应的entry位置
	addq	%rcx, %rax
	// 重新计算ghcb的物理地址，放在rsi之中，再给他加了个3
	// 还是很疑惑为什么要加3
    // 为了实现共享，这边就不再使用C-bit来做了，否则Hypervisor就没法看到
	subq	%rbx, %rsi
	addq	$0x03, %rsi
	movq	%rsi, (%rax)

	/* Replace the huge PMD entry with the new PTE */
	leaq	ghcb(%rip), %rsi
	movq	%rsi, %rax
	shrq	$PAGE_2MB_SHIFT, %rax
	andq	$PAGE_TABLE_INDEX_MASK, %rax
	// 根据ghcb的地址来确定PMD entry中对应的offset位置
	shlq	$3, %rax
	leaq	pmd(%rip), %rcx
	// 这下找到PTE应该存放的地方了
	addq	%rcx, %rax

	leaq	pte(%rip), %rdx
    // rdx现在存放着pte的物理地址
	subq	%rbx, %rdx
	movq	$SVSM_PTE_ENTRY(0), %rcx
	addq	%rcx, %rdx
    // 与rdx相加，放到合适的地方，即rax，当然这边也要加上C-bit
	movq	%rdx, (%rax)

	/* Flush the TLB - no globals, so CR3 update is enough */
	mov	%cr3, %rax
	mov	%rax, %cr3

	/* Zero out the early GHCB */
	cld
	leaq	ghcb(%rip), %rdi
	movq	$PAGE_SIZE, %rcx
	xorq	%rax, %rax
	// 复制al寄存器中的值，写入到rdi所指向的位置，写512个字节
	rep	stosb

	/* Zero out the BSS memory */
	cld
	leaq	sbss(%rip), %rdi
	leaq	ebss(%rip), %rcx
	subq	%rdi, %rcx
	xorq	%rax, %rax
	rep	stosb

	/* Save the start and end of the SVSM and dynamic memory */
	// guest virtual address start point -> svsm_begin -> svsm_gva_asm
	// svsm_end -> mem_size -> 256 MB
	movq	$SVSM_GVA_ASM, %rax
	movq	%rax, svsm_begin(%rip)
	addq	$SVSM_MEM_ASM, %rax
	movq	%rax, svsm_end(%rip)

	movq	%rax, dyn_mem_end(%rip)
	leaq	SVSM_DYN_MEM_BEGIN(%rip), %rax
	movq	%rax, dyn_mem_begin(%rip)

hl:
	xorq	%rdi, %rdi
	xorq	%rsi, %rsi
	xorq	%rdx, %rdx
	xorq	%rcx, %rcx
	xorq	%r8, %r8
	xorq	%r9, %r9

	movq	hl_main(%rip), %rax
	call	*%rax

	movq	$0x3f100, %rsi
	jmp	terminate_64

/*
 * 64-bit termination MSR protocol termination and HLT loop
 */
terminate_64:
	movq	%rsi, %rax
	movq	$0, %rdx
	movq	$0xc0010130, %rcx
	wrmsr
	rep;	vmmcall
terminate_hlt:
	hlt
	jmp	terminate_hlt

	.section .data
/*
 * Four zeroed stack pages with associated guard page.
 */
	.balign	4096
GLOBAL(bsp_guard_page)
	.fill	512, 8, 0
bsp_stack_start:
	.fill	512, 8, 0
	.fill	512, 8, 0
	.fill	512, 8, 0
	.fill	512, 8, 0
bsp_stack_end:

/*
 * 64-bit GDT.
 */
	.balign	8
GLOBAL(gdt64)
	.quad	0			/* Reserved */
kernel_cs:
	.quad	SVSM_KERNEL_CS_ATTR	/* 64-bit code segment (CPL0) */
kernel_ds:
	.quad	SVSM_KERNEL_DS_ATTR	/* 64-bit data segment (CPL0) */

tss:
	.quad	SVSM_TSS_ATTR0		/* 64-bit TSS */
	.quad	SVSM_TSS_ATTR1		/* TSS (Second half) */
GLOBAL(gdt64_end)

GLOBAL(gdt64_kernel_cs)
	.quad	SVSM_KERNEL_CS_SELECTOR

GLOBAL(gdt64_kernel_ds)
	.quad	SVSM_KERNEL_DS_SELECTOR

GLOBAL(gdt64_tss)
	.quad	SVSM_TSS_SELECTOR

GLOBAL(early_tss)
	.quad	tss

/*
 * 64-bit IDT - 256 16-byte entries
 */
	.balign 8
GLOBAL(idt64)
	.fill	2 * 256, 8, 0
GLOBAL(idt64_end)

/*
 * BSP/AP support:
 *   SMP support will update these values when starting an AP to provide
 *   information and unique values to each AP. This requires serialized
 *   AP startup.
 */
GLOBAL(cpu_mode)
	.long	0
GLOBAL(cpu_stack)
	.quad	bsp_stack_end
GLOBAL(cpu_start)
	.quad	code_64

/*
 * 64-bit identity-mapped pagetables:
 *   Maps only the size of the working memory of the SVSM.
 *   (e.g. 0x8000000000 - 0x800fffffff for 256MB)
 */
	.balign	4096
pgtables_start:
pgd:
	.fill	SVSM_PGD_INDEX, 8, 0
	.quad	SVSM_PGD_ENTRY(GPA(p4d))
	.fill	511 - SVSM_PGD_INDEX, 8, 0
p4d:
	.fill	SVSM_P4D_INDEX, 8, 0
	.quad	SVSM_P4D_ENTRY(GPA(pud))
	.fill	511 - SVSM_P4D_INDEX, 8, 0
pud:
	.fill	SVSM_PUD_INDEX, 8, 0
	.quad	SVSM_PUD_ENTRY(GPA(pmd))
	.fill	511 - SVSM_PUD_INDEX, 8, 0
pmd:
	.fill	SVSM_PMD_INDEX, 8, 0
	i = 0
	.rept	SVSM_PMD_COUNT
	.quad	SVSM_PMD_ENTRY(SVSM_GPA_ASM + i)
	i = i + SVSM_PMD_SIZE
	.endr
	.fill	511 - SVSM_PMD_INDEX - SVSM_PMD_COUNT + 1, 8, 0

/*
 * Reserve one extra page to split the 2MB private page that holds the
 * early GHCB so that a GHCB can be used for early page validation.
 */
pte:
	.fill	512, 8, 0
pgtables_end:

/*
 * Reserved an area for an early-usage GHCB, needed for fast validation
 * of memory.
 * ------------------------------------------------------------------
 * 很早在数据段开的一个ghcb空间，用户和虚拟机都可以访问，用于内存的合法性检验
 * 大小正好是4 KB，对应的恰好是可以由PVALIDATE指令进行操作的类型
 * 当然，我们一会儿还是得看一下页表是怎么生造出来的
 */
	.balign	4096
ghcb:
	.fill	512, 8, 0

/*
 * Main high-level language function to call
 */
GLOBAL(hl_main)
	.quad	svsm_main

/*
 * SEV related information
 */
GLOBAL(early_ghcb)
	.quad	0

GLOBAL(sev_encryption_mask)
	.quad	CBIT(0)

GLOBAL(sev_status)
	.quad	0

GLOBAL(svsm_begin)
	.quad	0

GLOBAL(svsm_end)
	.quad	0

GLOBAL(dyn_mem_begin)
	.quad	0

GLOBAL(dyn_mem_end)
	.quad	0

GLOBAL(svsm_sbss)
	.quad	sbss

GLOBAL(svsm_ebss)
	.quad	ebss

GLOBAL(svsm_sdata)
	.quad	sdata

GLOBAL(svsm_edata)
	.quad	edata

GLOBAL(guard_page)
	.quad	bsp_guard_page

GLOBAL(svsm_secrets_page)
	.quad	SVSM_SNP_SECRETS_PAGE_BASE

GLOBAL(svsm_secrets_page_size)
	.quad	SVSM_SNP_SECRETS_PAGE_SIZE

GLOBAL(svsm_cpuid_page)
	.quad	SVSM_SNP_CPUID_PAGE_BASE

GLOBAL(svsm_cpuid_page_size)
	.quad	SVSM_SNP_CPUID_PAGE_SIZE

GLOBAL(bios_vmsa_page)
	.quad	SVSM_SNP_BIOS_BSP_PAGE_BASE

/*
 * SVSM GUID Table
 */
	.section .data.guids

	/* Place the GUIDs at the end of the page */
	.balign	4096
	.fill	4096 - ((svsm_guids_end - svsm_fill_end) % 4096), 1, 0
svsm_fill_end:

/*
 * SVSM SEV SNP MetaData
 *   (similar to OVMF format, but addresses expanded to 8 bytes)
 */
svsm_snp_metadata:
	.byte	'S', 'V', 'S', 'M'					/* Signature */
	.long	svsm_snp_metadata_end - svsm_snp_metadata		/* Length */
	.long	1							/* Version */
	.long	(svsm_snp_metadata_end - svsm_snp_sections ) / 16	/* Section Count */

svsm_snp_sections:
	/* SEV SNP Secrets Page */
	.quad	GPA(SVSM_SNP_SECRETS_PAGE_BASE)
	.long	SVSM_SNP_SECRETS_PAGE_SIZE
	.long	2

	/* SEV SNP CPUID Page */
	.quad	GPA(SVSM_SNP_CPUID_PAGE_BASE)
	.long	SVSM_SNP_CPUID_PAGE_SIZE
	.long	3

	/* BIOS BSP VMSA Page */
	.quad	GPA(SVSM_SNP_BIOS_BSP_PAGE_BASE)
	.long	SVSM_SNP_BIOS_BSP_PAGE_SIZE
	.long	5
svsm_snp_metadata_end:

/*
 * SVSM GUID Envelope: 81384fea-ad48-4eb6-af4f-6ac49316df2b
 */
svsm_guids_start:

/* SVSM SEV SNP MetaData GUID: be30189b-ab44-4a97-82dd-ea813941047e */
svsm_guid_snp:
	.long	svsm_guids_end - svsm_snp_metadata			/* Offset to metadata */
	.word	svsm_guid_snp_end - svsm_guid_snp
	.byte	0x9b, 0x18, 0x30, 0xbe, 0x44, 0xab, 0x97, 0x4a
	.byte	0x82, 0xdd, 0xea, 0x81, 0x39, 0x41, 0x04, 0x7e
svsm_guid_snp_end:

/* SVSM INFO GUID: a789a612-0597-4c4b-a49f-cbb1fe9d1ddd */
svsm_guid_info:
	.quad	SVSM_GPA_ASM						/* SVSM load address */
	.quad	SVSM_MEM_ASM						/* SVSM memory footprint */
	.quad	CBIT(GPA(p4d))						/* SVSM pagetable (4-level) */
	.quad	gdt64							/* SVSM GDT */
	.word	SVSM_GDT_LIMIT						/* SVSM GDT limit */
	.quad	idt64							/* SVSM IDT */
	.word	SVSM_IDT_LIMIT						/* SVSM IDT limit */
	.word	SVSM_KERNEL_CS_SELECTOR					/* SVSM 64-bit CS slot */
	.quad	SVSM_KERNEL_CS_ATTR					/* SVSM 64-bit CS attributes */
	.quad	code_64							/* BSP start RIP */
	.quad	SVSM_EFER						/* SVSM EFER value */
	.quad	SVSM_CR0						/* SVSM CR0 value */
	.quad	SVSM_CR4						/* SVSM CR4 value */
	.word	svsm_guid_info_end - svsm_guid_info
	.byte	0x12, 0xa6, 0x89, 0xa7, 0x97, 0x05, 0x4b, 0x4c
	.byte	0xa4, 0x9f, 0xcb, 0xb1, 0xfe, 0x9d, 0x1d, 0xdd
svsm_guid_info_end:

	.word	svsm_guids_end - svsm_guids_start
	.byte	0xea, 0x4f, 0x38, 0x81, 0x48, 0xad, 0xb6, 0x4e
	.byte	0xaf, 0x4f, 0x6a, 0xc4, 0x93, 0x16, 0xdf, 0x2b
svsm_guids_end:

```

到这里，我们的分析便结束了，后面有很多变量是在`Rust`代码中使用的，特别提醒。