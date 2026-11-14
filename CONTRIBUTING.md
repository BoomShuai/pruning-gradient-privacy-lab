# 贡献说明

欢迎修复数学错误、增加可复现实验或改善文档。请先在 Issue 中说明具体问题与预期结果。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pruning_gradient_privacy run --output outputs/check --seed 42
```

数学变更应给出假设、推导和能识别错误的数值反例。实验变更应说明公开上下文、标签可见性、mask 生成规则、初始化、种子与预算口径。实际运行的测量与未来研究协议需要分别标明。

请不要提交个人信息、未经授权的题目、原始训练数据、API 密钥、模型权重或本地环境。运行输出通常放在 `outputs/` 并保持忽略。报告合成实验结果时附命令、Python/NumPy 版本及种子，不用单次优化失败推导隐私安全结论。

代码贡献按 MIT 发布，原创文字与图示按 CC BY 4.0 发布。第三方材料需要保留出处和许可证。使用生成式 AI 辅助贡献时，请披露其范围，并核验由此产生的代码、公式与引文。
