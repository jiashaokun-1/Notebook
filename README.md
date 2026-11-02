# Notebook

这是我的技术笔记仓，主要记录模型建模、并行策略、算子、编译优化、仿真方法和工程实践。

## 目录

- `Model/`：模型结构、图构建和代码走读。
- `parallel/`：推理并行策略与扩展性分析。
- `Operator/`：算子与编译相关记录。
- `Inductor/`：PyTorch Inductor 体系和算子融合。
- `Simulate/`：功能与性能仿真、指令解释和计算图提取。
- `Train/`：训练场景的数据采集和分析。
- `Testing/`：测试、探针和证据可信边界。
- `cleancode/`：代码规范和静态检查。

## 阅读入口

- [ELF、ISA 与 NPU-Sim：Ascend 算子如何在 Mac 上执行](Simulate/ELF、ISA与NPU-Sim：Ascend算子如何在Mac上执行.md)
- [大模型常用激活函数：作用、公式与曲线](Model/大模型常用激活函数：作用、公式与曲线.md)
- [常用通信算子：语义、数值示例与建模关系](parallel/常用通信算子：语义、数值示例与建模关系.md)
- [探针：用小实验验证局部行为](Testing/探针：用小实验验证局部行为.md)
- [MHA、GQA、MQA 与 MLA：结构、KV Cache 和 TP 切分](Model/Attention/MHA、GQA、MQA与MLA：结构、KV-Cache和TP切分.md)

笔记记录的是特定版本和特定条件下的判断依据。涉及外部项目时，应同时保存源码版本、实验输入、
观测结果和结论边界，避免把一次实验结果误当作普遍规律。
