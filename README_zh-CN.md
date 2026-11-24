# AgroWM

<div align="center">
<p>Built with Qwen</p>
<p><strong>先推理发育过程，再生成未来生长图像。</strong></p>
<p><a href="README.md">English</a> · <strong>简体中文</strong></p>
<p><a href="https://arxiv.org/abs/2611.26145">论文（arXiv）</a> · <a href="https://github.com/ECNU-ICALK/ArgoWM">GitHub</a> · <a href="https://huggingface.co/ecnu-icalk/ArgoWM">Hugging Face</a></p>
<p><a href="docs/INSTALL.md">安装</a> · <a href="docs/RUNNING.md">训练与推理</a> · <a href="docs/RESULTS.md">实验结果</a> · <a href="protocols/README.md">Prompt 与协议</a></p>
</div>

![AgroWM：发育推理、Growth Script 与图像生成](assets/method-overview.png)

*论文 Method 部分的 Figure 3 原图。图中样例相隔 49 天，水稻 prompt 中对应 7 周。[矢量 PDF](assets/method-overview.pdf)。*

## 模型简介

**AgroWM** 研究农业世界建模：从一张植物照片及预测间隔，预测**同一株植物**在数天至数月后的外观。**PhenoClock** 推断当前阶段、预测未来阶段并生成结构化的 **Growth Script**；**Rooted Renderer** 根据脚本编辑源照片，保留植物身份、视角和构图，并生成相应的器官、形态和颜色变化。

对应论文题目为 **[Can World Models Predict Rice Growth? Toward Agricultural World Models](https://arxiv.org/abs/2611.26145)**。水稻和玉米使用同一框架，分别从原始预训练模型训练。

| 项目 | 说明 |
|---|---|
| PhenoClock | 基于 Qwen3.8-27B 的 LoRA 微调 |
| Rooted Renderer | 基于 Qwen Image 2.1 的 diffusion transformer 全参数微调 |
| 中间表示 | Stage reasoning → Growth plan → Target phenotype |
| 水稻输入 | 源照片、品种/材料编号、正视或俯视、取整周数 |
| 玉米输入 | 源照片、植株 ID、侧视、精确天数 |
| 推理条件 | 不输入真实源阶段、目标阶段或目标照片 |
| 数值使用方式 | 作为脚本文本输入，不使用额外数值注入适配器 |
| 图像推理 | 20 步、BF16、固定的逐查询种子和原生尺寸 |
| 代码及 Planner adapter 许可证 | [Apache-2.0](LICENSE) |
| Renderer 权重许可证 | [Qwen Research License](LICENSES.md)，限非商业研究和评估 |

## 代码与模型版本对应

本次发布编号为 **`agrowm-mainline-2026-11-24`**。本目录是 GitHub 代码仓库；配套的 Hugging Face 模型仓库包含最终 Ours 权重及相同代码快照。两边使用同一套实现、协议、配置和共享文档，本代码包不分发权重。[两个仓库的使用方式](docs/REPOSITORIES.md)说明如何连接目录，[模型清单](configs/model_release.json)记录四个组件及其 SHA-256。

## 本仓库提供什么

本仓库包含主线代码、配置、prompt、schema、文档、格式示例和测试。**水稻和玉米的最终 AgroWM 权重由配套 Hugging Face 模型仓库提供**，不包含在本代码包内。每种作物各有一份 PhenoClock LoRA adapter 和一份 Rooted Renderer diffusion transformer 权重，共四份。文件结构、校验值与加载方式见[模型权重说明](docs/CHECKPOINTS.md)。

本次仅发布 Ours 的权重。Simple FT 等基线权重、优化器状态、数据集和原始基础模型文件不包含在仓库中。使用时需另行准备 Qwen3.8-27B 和 Qwen Image 2.1 基座，分别提供 Planner 骨干及 tokenizer/processor、Renderer 的冻结文本编码器和 VAE 等组件。

AgroWM 由推理模块和图像生成模块组成，使用源码目录与 editable installation；本仓库不能作为一个完整单模型目录直接通过 `from_pretrained()` 加载。内部包名 `wgsr_v9` 和协议 ID 保留用于兼容已有实验文件。

## 快速开始

使用 Python 3.12、Linux 和 NVIDIA GPU，Planner 与 Renderer 分别安装在独立环境中，见[安装说明](docs/INSTALL.md)。先按[仓库对应说明](docs/REPOSITORIES.md)将配套模型仓库的 `checkpoints/` 复制或链接到本目录，再另外准备两套原始基座模型。

在仓库根目录中配置已发布权重：

```bash
cp configs/released.example.json configs/local.released.json
# 修改基础模型、数据、Python 环境和输出路径。
python scripts/configure.py \
  --settings configs/local.released.json --output configs/local-released
```

示例配置已同时指向两种作物的 PhenoClock adapter 和 Rooted Renderer 权重。按[数据说明](docs/DATA.md)准备固定的成对样本、教师脚本和种子；导入包中的教师和目标信息不会进入推理请求。下方命令采用默认的 `run_root=runs`。若修改了 `run_root`，必须把 Planner 的 `--output-dir` 同步改为 `<你设置的 run_root>/$crop/planner_inference`，确保 Renderer 读取的是本次生成的脚本。随后运行：

```bash
crop=rice  # 或 maize

.venv-planner/bin/python scripts/launch_planner_inference.py \
  --config "configs/local-released/$crop/planner_infer.json" \
  --output-dir "runs/$crop/planner_inference" \
  --batch-size 1 --gpu-groups '0,1'

.venv-renderer/bin/python scripts/run_executor_inference.py \
  --config "configs/local-released/$crop/executor_infer.json" \
  --variant predicted_cot
```

生成的配置会自动选择对应作物的已发布 Renderer；`--checkpoint` 参数仍可用于显式覆盖。GPU 组合需能容纳 Planner。当前入口读取准备好的 manifest；未提供独立的原始单图推理便捷 API。无效脚本会记录为失败，不会自动替换为看过目标的 Tutor 脚本。

如需从基座重训，使用 `configs/local.example.json` 并按照[主线流程](docs/RUNNING.md)操作。水稻和玉米独立训练；Simple FT 也需要单独训练，其权重未随本次发布。

## Growth Script 格式

![水稻正视、俯视与玉米侧视的脚本格式](assets/growth-script-templates.png)

三部分分别回答：现在和未来处于什么阶段、需要发生哪些可量化变化、未来外观是什么样。水稻采用 S1–S8 阶段和随视角变化的字段，玉米采用 S1–S9 并增加雄穗和果穗描述。阶段词表包含在系统提示中，但不提供查询植株的真实阶段标签。

完整格式见[协议索引](protocols/README.md)，合成文本示例见 [examples](examples/README.md)。

## 论文结果

以下为论文报告值，**不是此次仓库整理重新测得的结果**。DINO、Stage R@1 越高越好；面积和高度误差以图像代理量的百分点表示，越低越好。

| 作物与方法 | DINO ↑ | 面积误差 ↓ | 高度误差 ↓ | Stage R@1 (%) ↑ |
|---|---:|---:|---:|---:|
| 水稻 · Qwen Image 2.1 FT | 0.834 | 3.73 | 7.83 | 47.4 |
| 水稻 · **AgroWM** | **0.860** | **3.51** | **7.43** | **54.6** |
| 玉米 · Qwen Image 2.1 FT | 0.764 | 1.97 | 11.12 | 30.1 |
| 玉米 · **AgroWM** | **0.795** | **1.73** | **10.61** | **30.9** |

在水稻 9–20 周预测中，AgroWM 的 Stage R@1 为 **46.4%**，Simple FT 为 **35.7%**，相差 **10.7 个百分点**。玉米的提升较小，为 **0.8 个百分点**，且不是从水稻模型直接零样本迁移的结果。

Stage R@1 是在同一株植物、同一视角的真实图像序列中检索目标阶段，不等于 Planner 的阶段分类准确率。其他基线、分母及比较条件见[结果说明](docs/RESULTS.md)。

## 数据与适用范围

| 项目 | 水稻 | 玉米 |
|---|---:|---:|
| 训练查询 | 20,000 | 6,400 |
| 验证查询 | 2,000 | 1,600 |
| 验证植株 | 79 | 84 |
| 图像尺寸 | 正视 672×896；俯视 896×672 | 侧视 768×1280 |
| 时间条件 | `max(1, (days + 3) // 7)` 周 | 精确天数 |

数据按植株划分；水稻验证集 79 个品种中有 78 个也出现在训练中。训练使用看过目标的教师脚本，正式推理只使用源照片预测脚本。背景处理和教师标注在本仓库的数据导入之前完成。

模型适用于发育推理、作物图像预测和表型评测研究。现有证据来自处理后的白背景水稻、玉米图像；田间场景、新品种和其他作物需要独立验证。天气、土壤和管理措施不是显式输入，生成图像也不能替代真实观测或产量测量。历史基线必须使用对应的原始提示、模型版本、种子和输出才能严格对比。

## 文档与许可证

- [模型权重与加载](docs/CHECKPOINTS.md)
- [安装与依赖](docs/INSTALL.md)
- [数据契约](docs/DATA.md)
- [训练、推理和评测](docs/RUNNING.md)
- [训练配置与指标](docs/PROTOCOL.md)
- [结果与比较范围](docs/RESULTS.md)
- [测试与开发](docs/DEVELOPMENT.md)
- [第三方依赖说明](THIRD_PARTY.md)

**代码、文档及 PhenoClock adapter 采用 Apache-2.0。Rooted Renderer 权重遵循 Qwen Research License，限非商业研究和评估，商业使用需要另行取得上游许可。** 各组件的许可范围见 [LICENSES.md](LICENSES.md)，署名见 [NOTICE](NOTICE)。第三方软件和数据遵循各自条款，本仓库不重新分发原始基座权重或数据集。
