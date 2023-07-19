## 有关于vc_init函数的一些分析
### 在这一节值得参考的代码
- 如何让在`GHCB`中更新的值被`Hypervisor`所知道 -> `rep vmmcall`
### vc_init: VC 中断处理初始化
`VC`问题本质上是`VMM communication exception`问题。这是由`CPU`触发的中断，一般是在需要`hypervisor emulation`环节时才会触发的中断。

可以查看`AMD-ES`和`AMD SEV-SNP`中的一部分流程图，从而得到一个基本的认识。在异常发生时，如`Guest`触发`VMEXIT`等相关指令时，`#VC`将会从硬件处发送给用户，让`vc_handler`与之相应，并对`GHCB`做一些简单的处理。

至于`vc`的初始化，我们可以看到其基本框架如下：
```rust
pub fn vc_init() {
    let ghcb_pa: PhysAddr = pgtable_va_to_pa(get_early_ghcb());

    vc_establish_protocol();
    vc_register_ghcb(ghcb_pa);
}
```
函数`pgtable_va_to_pa`示意图如下所示：
```rust
/// Obtain physical address (PA) of a page given its VA
pub fn pgtable_va_to_pa(va: VirtAddr) -> PhysAddr {
    PhysAddr::new_truncate(va.as_u64() - SVSM_GVA_OFFSET.as_u64())
}
```
似乎在`SVSM`中一开始的地址转换较为轻松，只需要把虚拟地址的高`12`位全部清除掉就好了。当然这是页表等东东尚未启动的时候的情况。不过在后续查看了以后，我们发现其实即便页表启用了也是这么做的（笑死）。
#### vc_establish_protocol
首先我们得到`ghcb_pa`的物理地址。接下来利用函数`vc_establish_protocol`来对`GHCB`模块按照协议进行处理。
```rust
fn vc_establish_protocol() {
    let mut response: u64;

    // Request SEV information
    // 询问SEV相关信息
    // GHCB_MSR_SEV_INFO_REQ是GHCB协议中专门用于REQ的位
    // 如果其值为2，则说明是这个功能，具体可以查看GHCB的参数表格。
    // 见2.3.1中的协议表格
    response = vc_msr_protocol(GHCB_MSR_SEV_INFO_REQ);

    // Validate the GHCB protocol version
    // 返回值须得是0x001，低十二位，检查合法与否。
    if GHCB_MSR_INFO!(response) != GHCB_MSR_SEV_INFO_RES {
        vc_terminate_ghcb_general();
    }

    // 检查response之中是否超出了大值和小值，支持的版本号是一串
    // 看样子是一串连续的数字，因此可以做这样的比较。
    if GHCB_MSR_PROTOCOL_MIN!(response) > GHCB_PROTOCOL_MAX
        || GHCB_MSR_PROTOCOL_MAX!(response) < GHCB_PROTOCOL_MIN
    {
        vc_terminate_ghcb_unsupported_protocol();
    }

    // Request hypervisor feature support
    // 询问Hypervisor相关feature支持
    // 0x80对应的功能，得到hypervisor feature support bitmap
    // 存放在GHCB INFO低十二位部分。
    // 高位被设置为0。
    response = vc_msr_protocol(GHCB_MSR_HV_FEATURE_REQ);

    // Validate required SVSM feature(s)
    // 理应得到的低12位返回值是0x81号数据，与上面相应成趣。作为返回值。
    if GHCB_MSR_INFO!(response) != GHCB_MSR_HV_FEATURE_RES {
        vc_terminate_ghcb_general();
    }

    // 对于返回的data值，检查其中的flag，必须要求其中的某一部分存在：
    // bit 0，bit 1两个部分是一定要存在的，其他无所谓
    // 不太清楚bit 4为什么也算在flag之中，我在GHCB中没有看到这个指示
    // 但是feature位本身就有52个bit，可能是有新的协议规定了bit 4作为SVSM的支持指示
    if (GHCB_MSR_HV_FEATURES!(response) & GHCB_SVSM_FEATURES) != GHCB_SVSM_FEATURES {
        vc_terminate_ghcb_feature();
    }

    // 将所支持的HV_FEATURES特性存储在这个名为HV的变量之中
    unsafe {
        HV_FEATURES = GHCB_MSR_HV_FEATURES!(response);
    }
}
```
函数`vc_msr_protocol`如下所示：
```rust
fn vc_msr_protocol(request: u64) -> u64 {
    let response: u64;

    // Save the current GHCB MSR value
    // 把当前的GHCB MSR值存放在value之中
    let value: u64 = rdmsr(MSR_GHCB);

    // Perform the MSR protocol
    // 将request写入MSR_GHCB寄存器，以求得到返回值，这些返回值会存放在寄存器之中，而寄存器信息会存放在GHCB之中
    wrmsr(MSR_GHCB, request);
    // rep vmmcall
    // 更新了MSR_GHCB的值之后，需要通过VMMCALL让hypervisor知晓，请与GHCB块进行交互，
    vc_vmgexit();
    response = rdmsr(MSR_GHCB);

    // Restore the GHCB MSR value
    wrmsr(MSR_GHCB, value);

    response
}
```
其中参数`request`的上文被设置为了`2`，恰好表示的意思是`REQUEST`相关。在这边利用指令rdmsr来讲`MSR_GHCB`固定需要投入的参数写入`rcx`寄存器，再把返回的结果写回到`value`之中，其中高位是`rdx`信息，低位是`rax`信息。

这一步的目的是把我们原本的值存放在`value`之中，然后我们把`request`信息写入到对应的寄存器中，希望得到返回值，之后呢再重新把原本值写回去，以执行我们后续的一些操作。

但函数`vc_vmgexit`相对来说比较简单，其中似乎主要就是`rep vmmcall`一下就好了。
```rust
fn vc_vmgexit() {
    unsafe {
        asm!("rep vmmcall");
    }
}
```
这行函数的目的是通知`Hypervisor`关于`GHCB`数据块信息的更新。

最终返回`response`作为结果。我们利用得到该信息后，需要检查这个`GHCB protocol`返回的信息是否是合法的。由于整个`flag`的大小是`12`位，需要与`0xfff`进行`&`操作。

根据手册上的指示，返回值一定是`0x001`才能说明奏效了。这便是第一步的检验。之后，针对第一步检验后得到的返回值`GHCBData`信息需要进行第二部检验。

根据手册上的信息，我们可以得知：
```
0x001 – SEV Information
▪ GHCBData[63:48] specifies the maximum GHCB protocol version supported.
▪ GHCBData[47:32] specifies the minimum GHCB protocol version supported.
▪ GHCBData[31:24] specifies the SEV page table encryption bit number.
```
用户需要通过`GHCBData`中提供的版本号来确定`hypervisor`到底支持哪些版本，从而选择合适版本并对`GHCB Protocol`协议的版本做一些限定。如果`guest`没法支持`hypervisor`提供的协议范围，那就寄了，需要发送`0x100`表示终结我们的任务。
#### vc_register_ghcb
在完成了`vc_establish_protocol`函数之后，我们进入函数`vc_register_ghcb`之中。该函数的大体情况如下所示：
```rust
pub fn vc_register_ghcb(pa: PhysAddr) {
    // Perform GHCB registration
    // 在用户初次使用GHCB之前，需要Register GHCB GPA信息，实现注册
    // pa | 0x12. -> ghcb pa | 0x12
    // only supported and required for SEV-SNP guests
    // DATA段会被设置成pa的GFN部分，即底部12位自动忽略掉
    // 恰好这个底部12位被设置成0x12.
    let response: u64 = vc_msr_protocol(GHCB_MSR_REGISTER_GHCB!(pa.as_u64()));

    // Validate the response
    // 按照手册，这一步很自然。
    // info部分需要被返回一个0x13
    if GHCB_MSR_INFO!(response) != GHCB_MSR_REGISTER_GHCB_RES {
        vc_terminate_svsm_general();
    }

    // 这一步也相当自然
    // Data部分需要被返回一个同样的GFN值
    if GHCB_MSR_DATA!(response) != pa.as_u64() {
        vc_terminate_svsm_general();
    }

    // 不知道为什么，反正先把pa写到了rax和rcx之中。
    // 可能之后有用到。
    wrmsr(MSR_GHCB, pa.as_u64());
}
```
我们一点点来看，先分析`vc_msr_protocol`的参数，其根据`ghcb_pa`，即`early_ghcb`的物理地址来确定它的`request`，其中`GHCB_MSR_REGISTER_GHCB`宏的信息如下所示：
```rust
// MSR protocol: GHCB registration
/// 0x12
const GHCB_MSR_REGISTER_GHCB_REQ: u64 = 0x12;
macro_rules! GHCB_MSR_REGISTER_GHCB {
    ($x: expr) => {
        (($x) | GHCB_MSR_REGISTER_GHCB_REQ)
    };
}
```
在这边本质上是把`Request`与`pa`一起封装成`64`位的协议数据包，可以参考手册，其上恰是这么要求的。
```
 0x012 – Register GHCB GPA Request
▪ GHCBData[63:12] – GHCB GFN to register
Written by the guest to request the GHCB guest physical address (GHCB GPA 
= GHCB GFN << 12) be registered for the vCPU invoking the VMGEXIT. See
section 2.3.2 for further details and restrictions.
```
在2.3.2环节中，这一段代码对应的是后面的`Register`环节。把这个信息写入到`request`变量中后，根据手册，我们需要看到`0x13`作为`GHCBInfo`返回值。而`data`段`hypervisor`则需要用`GHCB GPA`所对应的`GFN`来进行响应。

到这一步，`GHCB`的`Register`工作完成了，之后用户就可以使用这个`GHCB`。

到这里，这个函数也分析完了。我们简单总结一下这个函数干了什么：

### 询问`SEV`是否被支持、询问`SEV features`是否支持`SVSM`、完成一次`GHCB`信息的基本传递、注册`GHCB`块以供后续用户使用。