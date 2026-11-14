# Pruning Gradient Privacy Lab

用 NumPy 探索 **Taylor 剪枝、Wanda 分数与稀疏梯度重建**，附中文理论研究笔记和可复现实验协议。

A small, CPU-only NumPy lab for pruning sensitivity and synthetic gradient reconstruction, with Chinese research notes. No model weights or datasets are downloaded.

## 可以运行什么

| 模块 | 实现内容 | 如何理解结果 |
| --- | --- | --- |
| Taylor 剪枝 | 二次损失的真实变化、一阶近似、Lipschitz 余项界与驻点反例 | 数值校验明确假设下的局部公式 |
| Wanda | 单权重剪枝的层输出扰动范数、与损失敏感度的反例 | Wanda 精确描述局部输出扰动；一般任务损失还取决于反向梯度 |
| 合成梯度重建 | 单样本线性回归，Taylor/梯度幅值/随机嵌套掩码，保留率与优化迭代次数扫描 | 比较当前优化器在一个小模型上的行为 |
| 理论研究笔记 | 条件 MDL、Epiplexity、Jacobian 可识别性与建议实验协议 | 形式化假设与待检验方案；未进行大模型实验 |

**当前实验不测量 Epiplexity、Shannon 熵、差分隐私或真实图像重建质量。** 小规模重建失败也不构成隐私保证。完整实现范围见 [实验说明](docs/experiments.md)。

## 安装与运行

需要 Python 3.10 或更新版本，仅依赖 NumPy。下列命令在仓库根目录执行：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pruning_gradient_privacy run --output outputs/demo --seed 42
```

也可以使用安装后的命令：

```bash
pruning-gradient-privacy run --output outputs/demo --seed 42
```

默认运行小规模合成实验，输出：

- `results.json`：配置、数学校验、解析恢复对照与重建结果。
- `inversion.csv`：方便再次分析的逐条件重建结果。
- `report.md`：可阅读的运行报告。
- `reconstruction.svg`：无需绘图库即可打开的结果图。

仓库提供一份真实运行生成的 [seed 42 示例报告](examples/seed42/report.md) 与 [SVG 结果图](examples/seed42/reconstruction.svg)，对应配置记录在 [results.json](examples/seed42/results.json) 中。

新运行结果建议放在已被 Git 忽略的 `outputs/` 中；`examples/seed42/` 是随仓库提交的参考示例。相同配置和种子产生可重复的结果；跨 NumPy 版本或硬件的浮点运算可能存在微小差异。

自定义实验：

```bash
python -m pruning_gradient_privacy run --output outputs/custom --seed 7 --dimension 8 --trials 5 --rhos 0,0.25,0.5,1 --budgets 0,25,100
python -m unittest discover -s tests -v
```

`--budgets` 表示 Adam 更新次数，**没有按 FLOPs 对齐**；不同保留率的每步计算成本可能不同。各条件共享样本、初始化与排序规则，随机掩码取同一随机排列的前缀。`rho=0` 是无梯度观测基线。

## 威胁模型与边界

合成实验向重建者公开模型、目标值和保留坐标，匹配所保留的权重梯度与偏置梯度。默认重建优化器使用固定支持上的数值误差，未利用“这些坐标为何进入 Top-k”的排序约束，因此没有完整审计支持集侧信道。完整梯度还提供解析恢复对照：偏置梯度非零时，可从权重梯度与偏置梯度直接恢复输入。

研究笔记中的 `T` 是建议研究使用的计算预算，而当前 CLI 使用迭代次数；两者不能直接互换。CIFAR-10/ResNet、DLG/iDLG、条件编码器与两部码/prequential 码长属于后续研究协议，尚未实现。

## 文档与来源

- [数学与实验定义](docs/experiments.md)
- [公开研究笔记源码](paper/notes.tex) · [编译说明](paper/README.md)
- [内容来源与参考文献核对范围](docs/provenance.md)
- [v1.0.0 验证记录](docs/validation.md)
- [数据边界](docs/data-boundaries.md) · [贡献说明](CONTRIBUTING.md)

该项目从既有理论稿整理而来，增加了独立实现的数值验证与合成实验。生成式 AI 参与了文档整理、代码开发和审查；测试及示例只支持其明确覆盖的行为。

## 许可证与引用

代码使用 [MIT](LICENSE)。原创文档、研究笔记及示意图使用 [CC BY 4.0](docs/LICENSE-CC-BY-4.0.md)。参考文献指向第三方作品，其原有权利不因本项目许可证改变。引用本项目请使用 [CITATION.cff](CITATION.cff)。
