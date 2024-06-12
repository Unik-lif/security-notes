## 应该干什么？
通过`Guest-OS`的`Linux`相关源码进行阅读，从而得知其对与`SVSM`的支持表现的怎么样。

首先在文件`sev.h`之中我们可以看到针对`secrets_page`有下面的布置：
```C
/* See the SNP spec version 1.51 for secrets page format */
struct snp_secrets_page_layout {
	u32 version;
	u32 imien	: 1,
	    rsvd1	: 31;
	u32 fms;
	u32 rsvd2;
	u8 gosvw[16];
	u8 vmpck0[VMPCK_KEY_LEN];
	u8 vmpck1[VMPCK_KEY_LEN];
	u8 vmpck2[VMPCK_KEY_LEN];
	u8 vmpck3[VMPCK_KEY_LEN];
	struct secrets_os_area os_area;

	u8 vmsa_tweak_bitmap[64];

	/* SVSM fields */
	u64 svsm_base;
	u64 svsm_size;
	u64 svsm_caa;
	u32 svsm_max_version;
	u8 svsm_guest_vmpl;
	u8 rsvd3[3];

	/* Remainder of page */
	u8 rsvd4[3744];
} __packed;

/*
 * The SVSM CAA related structures.
 */
 // 这一部分具体怎么用或许可以参考SVSM Draft
 // 涉及calling address area，是客户用于调用的函数锁放置的地方
struct svsm_caa {
	u8 call_pending;
	u8 mem_available;
	u8 rsvd1[6];

	u8 svsm_buffer[PAGE_SIZE - 8];
};
```
对于`pvalidate operation`，在源码中有这样的设置。
```C
/*
 * The SVSM PVALIDATE related structures
 */
struct svsm_pvalidate_entry {
	u64 page_size		: 2,
	    action		: 1,
	    ignore_cf		: 1,
	    rsvd		: 8,
	    pfn			: 52;
};

struct svsm_pvalidate_call {
	u16 entries;
	u16 next;

	u8 rsvd1[4];

	struct svsm_pvalidate_entry entry[]; // 最少有一个，最多最好不能超过4 KB的页边界，不然会比较麻烦
};
```