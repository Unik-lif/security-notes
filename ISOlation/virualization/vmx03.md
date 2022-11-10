## VT-X
### Trap-and-emulate
[参考文章](https://lettieri.iet.unipi.it/virtualization/2018/hardware-assisted-intel-vmx.pdf)

在上一则文章中我们简要介绍了`Trap-and-emulate`的工作流程，并举例了部分需要进行`Trap`的指令。简而言之，最终效果是让我们的`target-machine system software`在用户层的权限中运行，在遇到了需要高权限的指令时，我们利用`Trap`将控制权交给`VMM`，并人`VMM`在虚拟机中模拟运行需要高权限指令的效果。

这种虚拟机仅依赖Intel x86架构是不能实现的，因为其存在部分指令在用户层运行时，哪怕用户层没有该高级权限，系统也不会发出`exception`信息，导致`VMM`想要模拟时无从知晓（没有信号），进而无法实现`Target machine`和`Virtual machine`的统一（`Target machine`是虚拟机希望展现出来的效果，最好和实际在物理机里运行保持一致，`Virtual machine`是我们真实移植起来的效果）。

以`popf`指令为例，该指令将会把一个两个字长的信息装入栈，并存放到`EFLAGS`寄存器中。对`EFLAGS`中的`IF`标志进行修改将会导致处理器外部的中断被失效。在我们预期中，用户态下运行`popf`指令时，`IF`标志不会出现更新。对于虚拟机来说，其目标是让处理器外部中断失效（虽然它现在是用户态运行之没有实现），但这件尝试举动理应让`VMM`知道，因此机器的目标效果和实际效果出现了分离。

此外还有类似于`%cr3`这样能在用户态阅读却无法修改的寄存器，虚拟机内的软件可以通过比较阅读`%cr3`内的信息（真实值）和自己尝试写入`%cr3`内的信息（该值的写入会造成`exception`，也就是需要通过`VMM`来办）意识到自己并不是在一个真实机器（或者目标机器）上运行。

因此，如果要让intel-x86机器能够支持虚拟化，需要扩展一些虚拟化组件。这就是VT-X的由来。

### VMX
`VMX`是`Virtual Machine eXtensions`的简称。其为Intel CPU提供了新的两种运作模式，`root mode`和`non-root mode`。特别的，其与`system mode`和`user mode`正交。

在`system code`尝试执行一些违背`VMM`隔离机制或者需要`trap`时，`trap`后需要递交控制权给`VMM`。

进入`non-root mode`：采用`VMLAUNCH`和`VMRESUME`，执行虚拟机内的代码。

进入`root mode`：采用`VM exits`，脱离虚拟机内的代码，让控制权转交给`VMM`。

所有新的`VM instructions`均仅仅在`root/system`下才能被允许执行。

### VMCS:
`VMCS`是`Virtual Machine Control Structure`的简称。

在`VMM`之中可能存在多个`VMCS`，基本上是一个处理器分配一个虚拟机。但是，仅有一个`VMCS`会被设置成`current`并加以运作。（参考`virtual01.md`，那么久不再赘述了）。

里头存放了一些信息：
- guest state:
- host state:
- VM execution control：规定哪些能做哪些不能做。
- VM enter control：包含`root`向`non-root`传递的一些可选行为。
- VM exit control：包含`non-root`向`root`传递的一些可选行为。
- VM exit reason：记录退出原因。

Guest state存放的部分信息：
- `%cr3`：页表指针
- `%idtr`：中断描述符表
- `%gdtr`：指向`global descriptor table`的指针
- `%tr`：selector of the current task
- 指令指针

