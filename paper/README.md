# 理论研究笔记

`notes.tex` 是独立整理的中文双栏研究笔记，包含 Taylor 剪枝、Wanda、条件 MDL/Epiplexity 分析，以及尚未执行的完整实验协议。它不包含真实个人数据或原始任务材料。

仓库可执行示例与研究协议的覆盖范围不同，详见 [实验说明](../docs/experiments.md)。本稿未经过独立同行评审；条件结论须连同其假设阅读。

## 使用 XeLaTeX 编译

需要包含 `latexmk`、XeLaTeX、BibTeX、IEEEtran、xeCJK、TikZ、TeX Gyre 和 Fandol 字体的 TeX Live 环境。字体使用 TeX Live 提供的 TeX Gyre Termes 与 Fandol 字体，避免依赖 macOS 系统字体。

在仓库根目录执行：

```bash
mkdir -p build/paper
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error -outdir=../build/paper paper/notes.tex
```

输出位于 `build/paper/notes.pdf`；构建目录已被 Git 忽略。图示源文件为 `figures/framework.tikz`，参考文献为 `references.bib`。代码 CI 不安装 TeX Live，也不代表此文稿已通过排版或数学证明的完整审计。

也可使用 Tectonic 0.17.0（首次构建会联网下载 TeX 依赖）：

```bash
mkdir -p build/paper
tectonic --outdir build/paper --keep-logs paper/notes.tex
```

公开版已用 Tectonic 0.17.0 完成编译，并检查了全部 7 页。编译后的脱敏 PDF 随 [v1.0.0 Release](https://github.com/BoomShuai/pruning-gradient-privacy-lab/releases/tag/v1.0.0) 提供。数值代码与文稿检查范围见 [验证记录](../docs/validation.md)。

## 许可证

原创文字与图示按 [CC BY 4.0](../docs/LICENSE-CC-BY-4.0.md) 发布。第三方参考文献和 TeX 依赖遵循其各自许可证。生成式 AI 参与文档整理与审查，该使用方式已在文稿中披露。
