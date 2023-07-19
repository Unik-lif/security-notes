## 有关idt_init函数的笔记整理：
### 在这一节值得参考的代码
### 中断处理：idt_init干了什么
`x86`一般处理中断的流程，是把中断向量表直接加载进来给特定硬件来使用。
```rust
/// Load IDT with function handlers for each exception
pub fn idt_init() {
    IDT.load();
}
```
我们先来看看`IDT`是什么东西。
```rust
lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt: InterruptDescriptorTable = InterruptDescriptorTable::new();

        unsafe {
            idt.double_fault
                .set_handler_fn(df_handler)
                .set_stack_index(DOUBLE_FAULT_IST as u16);
        }
        idt.general_protection_fault.set_handler_fn(gp_handler);
        idt.page_fault.set_handler_fn(pf_handler);
        idt.vmm_communication_exception.set_handler_fn(vc_handler);

        idt
    };
}
```
简单看一下结构体`InterruptDescriptorTabele`，这一版本的中断处理表是由`AMD`特别修改过的。其中前`32`个`entries`是用于处理`CPU`异常的，而后面的则是用于处理中断。每个中断之中都存放了处理函数的位置，以便进行索引并处理对应的相关异常问题。这个中断处理表在初始化的时候一共有`256`个项。

我们可以看到这里用了很经典的`lazy_static`方法，并利用`x86-64`模块中的部分组件来实现，当然，他们使用的模块是经过自己修改后的版本，内部嵌入了关于`#VC`、`#GP`等错误处理器放置的地方。在`unsafe`模块中设置了`df_handler`作为`double_fault`的处理者。这边的处理流程似乎都相对简单，由于对应的是异常，只需要报错`panic`就可以了。此外针对`#VC`的情况，还需要走`terminate`的路子。

其他异常处理采用的初始化方式类似。最后利用`lidt`指令装载进去，即可完成中断处理表的设置工作。

### 总结，这一步主要做了对于中断与异常处理表的设置工作。