# 内容来源与公开范围

本项目由一份既有中文理论稿整理而来，保留其技术分析、参考文献与方法框图，移除了机构标识、任务题面及个人身份栏。代码为本仓库新增的独立 NumPy 实现，没有复制下列论文的官方实现。

公开内容包括：

- 经过整理的理论稿及可编辑 TikZ 图；
- Taylor/Wanda 数值验证；
- 合成线性回归的梯度匹配实验与运行说明；
- 测试、CI、许可证和研究边界说明。

原始题面、个人材料、校徽、原始 PDF、压缩包、编译日志均不属于公开项目。原始材料的私人备份与公开仓库分开管理。

## 已核对的关键参考文献

以下链接为论文原始页面。已核对这四项的身份和主要书目信息；没有对所有参考文献进行全文事实核查，也没有验证每条引文是否足以支持原稿中的每个推论。

1. Molchanov et al., *Importance Estimation for Neural Network Pruning*, CVPR 2019, pp. 11264–11272. [CVF 官方页面](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)。公开书目已纠正原稿页码。
2. Sun et al., *A Simple and Effective Pruning Approach for Large Language Models*, ICLR 2024. [作者预印本](https://arxiv.org/abs/2306.11695)。本项目不复现完整大语言模型剪枝流程。
3. Zhu et al., *Deep Leakage from Gradients*, NeurIPS 2019. [作者预印本](https://arxiv.org/abs/1906.08935)。本项目的线性梯度匹配示例不是 DLG 的完整复现。
4. Finzi et al., *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence*, 2026. [预印本 v2](https://arxiv.org/abs/2601.03220v2)。本项目保留相关理论讨论，未实现其程序搜索或码长估计。

完整书目见 [references.bib](../paper/references.bib)。引用第三方论文不表示作者认可本项目，也不把这些工作的创新归属于本项目。
