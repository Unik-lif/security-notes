## 梦开始的地方
启动时使用的脚本如下所示：
```shell
#!/bin/bash
./qemu/build/qemu-system-riscv64 -cpu rv64 -M virt -m 1024M -nographic -bios opensbi/build/platform/generic/firmware/fw_jump.bin -kernel ./build-riscv64/arch/riscv/boot/Image -initrd ./rootfs_kvm_riscv64.img -append "root=/dev/ram rw console=ttyS0 earlycon=sbi"
```
可以看出使用了`riscv64`的`qemu`来启动了一个作为`host`的机器，这边似乎依赖了`opensbi`，以及`image`。

看起来除了`opensbi`和`linux`被修改了，其他应该都是一样的。

因此，我们需要先搞清楚这边的启动方式是怎么跑起来的，也就是先搞清楚各个组件是怎么工作的，然后再考虑下一步。

根据qemu手册，启动时使用的`-bios`指向的就是机器的firmware固件，是加电之后先跑的程序，指向opensbi，之后再链接linux对应的kernel文件。

一开始使用的文件系统，即`-initrd`指向的位置就是初始化后的文件系统位置，将会用`rootfs_kvm_riscv64.img`作为文件系统镜像，而这个则是可以由busybox所构造出来的。

我们目前阅读完了所有设备的编译流程，现在对于做过修改的部分做一些自己的解读。
## opensbi
首先我们注意到是以`generic`模式启动，搜索关键词`fw_jump.bin`，进入到相关的文档中。
- **fw_jump.md**: 告知fw_jump文件的大体作用，即跳转到操作系统内核中。主要还是指示`qemu`在经过`opensbi`之后，下一阶段的启动应该跳转到哪个位置上。
- **qemu_virt.md**: 基本使用方法，包括如何利用GDB进行过debug。

Makefile的生成过程：通过`Makefile`的`make -nB`进行分析
- 首先会把sbi文件夹中的全部文件拿过来，生成一个叫做`lib/libsbi.a`的静态库。
- 根据先前得到的`generic`平台信息，生成一些`platform-specific`，和先前sbi生成的.o文件一起，组成一个叫做`libplatsbi.a`的东西。
- 通过test相关文件，并上先前的`libplatsbi.a`得到新的`test.elf`文件，并进一步得到`test.bin`文件。
- test.bin fw_dynamic.S => fw_dynamic.o, test.bin as the DFW_PAYLOAD.
- fw_dynamic.o fw_jump.elf.ld => fw_jump.o
- fw_jump.o libplatsbi.a => fw_jump.elf => fw_jump.bin


关键语句：
```shell
mkdir -p `dirname /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.elf`; echo " ELF       platform/generic/firmware/fw_jump.elf"; riscv64-linux-gnu-gcc -g -Wall -Werror -ffreestanding -nostdlib -fno-stack-protector -fno-strict-aliasing -O2 -fno-omit-frame-pointer -fno-optimize-sibling-calls -mno-save-restore -mstrict-align -mabi=lp64 -march=rv64imafdc_zicsr_zifencei -mcmodel=medany  -I/home/link/Desktop/riscv-kvm/opensbi/platform/generic/include -I/home/link/Desktop/riscv-kvm/opensbi/include -w -include /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/kconfig/autoconf.h -I/home/link/Desktop/riscv-kvm/opensbi/lib/utils/libfdt/  -DFW_JUMP_OFFSET=0x200000 -DFW_JUMP_FDT_OFFSET=0x2200000 -DFW_PAYLOAD_PATH=\"/home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/payloads/test.bin\" -DFW_PAYLOAD_OFFSET=0x200000 -DFW_PAYLOAD_FDT_OFFSET=0x2200000  -fPIE -pie  /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.o /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/lib/libplatsbi.a -fuse-ld=bfd -Wl,--build-id=none -Wl,--no-dynamic-linker -Wl,-pie   -Wl,-T/home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.elf.ld -o /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.elf
mkdir -p `dirname /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.bin`; echo " OBJCOPY   platform/generic/firmware/fw_jump.bin"; riscv64-linux-gnu-objcopy -S -O binary /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.elf /home/link/Desktop/riscv-kvm/opensbi/build/platform/generic/firmware/fw_jump.bin
```

### 起始文件
从上述来看，我们直接研究`fw_jump.o`开始就行。

阅读研究这边的资料很有帮助：

https://blog.csdn.net/zyhse/article/details/138545606