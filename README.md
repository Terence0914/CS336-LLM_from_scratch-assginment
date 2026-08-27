# CS336 Assignment 1 with Modern LLM Extensions

> Temporary bilingual README / 暂时性双语说明

This is an independent educational implementation based on Stanford University's **CS336: Language Modeling from Scratch — Assignment 1: Basics**. It is not an official Stanford repository and is not affiliated with Stanford University, Moonshot AI, or DeepSeek.

本项目是基于斯坦福大学 **CS336：Language Modeling from Scratch — Assignment 1: Basics** 的个人学习与研究实现，并非斯坦福官方仓库，也不代表斯坦福大学、Moonshot AI 或 DeepSeek。

---

## 中文说明

### 项目简介

本项目以 Stanford CS336 Assignment 1 为基础，从零实现语言模型的核心组件，包括：

- BPE tokenizer 与文本预处理；
- Transformer 基础模块；
- RMSNorm、SwiGLU 与 RoPE；
- scaled dot-product attention 与多头注意力；
- 交叉熵损失、AdamW、学习率调度和训练循环；
- TinyStories / OpenWebText 上的训练与评估。

当前仓库正在逐步完成 Assignment 1；其中 scaled dot-product attention 已通过 Stanford CS336 提供的三阶与四阶输入测试，并通过布尔 mask 的本地数值验证。

### 与官方作业的关系

Stanford CS336 Assignment 1 的核心目标是实现一个标准 Transformer 语言模型及其完整训练流程。课程会讲授 Mixture-of-Experts，但官方 Assignment 1 并未要求复现 DeepSeekMoE。因此，本项目把 DeepSeekMoE 作为独立于原作业要求的研究扩展。

### 个人研究扩展：复现与组合创新

#### 1. Moonshot AI Attention Residuals（计划中）

计划复现 Moonshot AI / Kimi Team 提出的 **Attention Residuals（AttnRes）**，探索用跨深度的可学习注意力聚合替代固定的残差累加：

- **Full AttnRes**：每层选择性聚合此前各层的表示；
- **Block AttnRes**：在块内保留普通残差，在块间使用注意力聚合，以降低深度方向的内存开销；
- 与标准 PreNorm residual baseline 比较训练稳定性、验证损失、梯度分布和计算开销。

#### 2. DeepSeekMoE（计划中）

由于官方 Assignment 1 没有实现 DeepSeekMoE，本项目计划将稠密 SwiGLU FFN 扩展为小规模 **DeepSeekMoE-style** 模块，重点复现：

- 细粒度专家划分（fine-grained expert segmentation）；
- 共享专家隔离（shared expert isolation）；
- Top-k 路由与激活参数量控制；
- 路由负载、专家利用率和训练稳定性的监控。

该实现将针对教学和有限算力环境缩小模型与专家规模，不声称复现 DeepSeek 原始训练规模或结果。

DeepSeekMoE 保留为独立的预训练架构探索，不进入本项目第一阶段的后训练对比。后训练与微调将固定使用 **Dense Transformer + AttnRes**，避免同时改变基础架构和训练算法而混淆实验结论。

#### 3. 轻量化 OPSD 后训练（计划中）

后训练阶段计划进行一个资源受限的 **On-Policy Self-Distillation（OPSD）** 实验：同一个 Dense Transformer + AttnRes 模型分别作为 student 与带有参考答案等 privileged context 的 teacher，在 student 自己生成的轨迹上进行 token-level 分布对齐。

这里的 OPSD 是 OPD 的单模型自蒸馏变体，而不只是按比例缩小 OPD。项目中的“轻量化”主要指：

- 使用更小的模型和数据集；
- 缩短 rollout 与训练周期；
- 优先实验 KL/JSD 等 token-level 蒸馏目标；
- 评估准确率、生成 token 成本、显存和训练稳定性。

### 中文数据与后训练方案

后训练数据固定到 Hugging Face 上 [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/6b952cc50427c84eac543d0b38a8066208433847) 的 commit `6b952cc50427c84eac543d0b38a8066208433847`，确保实验可以复现。该数据仓库以中文和多语言生成数据为主，并提供预训练、SFT、推理蒸馏和偏好数据。

计划按用途划分数据：

- `sft_mini_512.jsonl`：作为主要中文 SFT 数据，先让模型获得稳定的指令遵循和对话格式；
- `r1_mix_1024.jsonl`：经清洗后作为 OPSD 的问题与参考解答来源；
- 从 `r1_mix_1024.jsonl` 中进一步筛选可自动验证最终答案的数学/代码样本，作为 GRPO 数据；
- `pretrain_hq.jsonl`：仅在需要中文领域继续预训练时使用，不与后训练结果混为一谈；
- `dpo.jsonl` 和 `rlaif-mini.jsonl`：不进入第一阶段的 OPSD–GRPO 主对比，可留作后续偏好优化实验。

使用前将进行去重、格式检查、中文比例过滤、答案可验证性检查和 train/validation/test 隔离。该仓库由多个上游数据源整合而来，并同时标注 Apache-2.0 与 CC-BY-NC-2.0；本项目将保留来源与 commit 信息，按更严格的非商业限制处理，并在发布派生数据前再次核查各文件许可。

### 实验路线：SFT + OPSD 对比 GRPO

第一阶段只使用同一个 **Dense Transformer + AttnRes** 预训练 checkpoint。完成中文 SFT 后，从完全相同的 SFT checkpoint 分叉，控制相同的训练 prompts、rollout 上限和生成 token 预算：

1. **SFT baseline**：仅进行中文监督微调；
2. **SFT → OPSD**：在 student 的 on-policy 轨迹上进行 token-level KL/JSD 自蒸馏；
3. **SFT → GRPO**：仅在具有自动 verifier 的同源数学/代码子集上进行结果奖励优化。

主要比较中文指令遵循、可验证任务 exact match / pass@1、生成 token 成本、峰值显存、训练时间、KL/JSD、GRPO reward 和训练稳定性。SFT-only 组用于测量两种后训练方法的真实增益，OPSD 与 GRPO 使用相同起点和尽可能一致的生成预算。所有结果都将在实际运行后再补充，本 README 不预先声明尚未得到的提升。

### 当前目录

```text
.
├── model.py          # Transformer 基础模块与 attention
├── adapters.py       # 测试适配入口
├── tokenizer.py      # Tokenizer
├── train_bpe.py      # BPE 训练
└── test/             # 本地测试与 smoke tests
```

### 本地环境

本地开发使用 Conda 环境 `ai_workspace`，并兼容 Colab 中的 `cs336_basics` 包结构。

```bash
conda activate ai_workspace
python -m pytest
```

---

## English

### Overview

This project builds on Stanford CS336 Assignment 1 and implements the core components of a language model from scratch:

- BPE tokenization and text preprocessing;
- foundational Transformer modules;
- RMSNorm, SwiGLU, and RoPE;
- scaled dot-product attention and multi-head attention;
- cross-entropy loss, AdamW, learning-rate scheduling, and the training loop;
- training and evaluation on TinyStories / OpenWebText.

The repository is a work in progress. The scaled dot-product attention implementation has passed the Stanford CS336 tests for third- and fourth-order tensors and has also been locally validated with a boolean mask.

### Relationship to the official assignment

The official CS336 Assignment 1 focuses on implementing a standard Transformer language model and its end-to-end training pipeline. Although the course covers Mixture-of-Experts, Assignment 1 does not require a DeepSeekMoE reproduction. DeepSeekMoE is therefore treated here as a separate research extension.

### Personal research extensions: reproduction and integration

#### 1. Moonshot AI Attention Residuals (planned)

The project plans to reproduce **Attention Residuals (AttnRes)** from Moonshot AI / the Kimi Team, replacing fixed residual accumulation with learned, input-dependent aggregation over depth:

- **Full AttnRes**, which selectively aggregates previous layer representations;
- **Block AttnRes**, which retains ordinary residuals inside blocks and performs attention across block summaries;
- comparison against the standard PreNorm residual baseline in stability, validation loss, gradient distribution, memory, and compute.

#### 2. DeepSeekMoE (planned)

Because DeepSeekMoE is not part of the official Assignment 1 implementation, this project plans a small-scale **DeepSeekMoE-style** replacement for the dense SwiGLU FFN, focusing on:

- fine-grained expert segmentation;
- shared expert isolation;
- top-k routing and controlled active parameters;
- monitoring routing balance, expert utilization, and training stability.

The implementation will be scaled to educational and limited-compute settings; it does not claim to reproduce DeepSeek's original training scale or published results.

DeepSeekMoE remains a separate pretraining architecture study and is excluded from the first post-training comparison. Fine-tuning and post-training will use a fixed **Dense Transformer + AttnRes** architecture so that architectural changes are not confounded with training-algorithm changes.

#### 3. Lightweight OPSD post-training (planned)

The post-training roadmap includes a resource-conscious **On-Policy Self-Distillation (OPSD)** experiment on the Dense Transformer + AttnRes model. A single model acts as both student and teacher under different contexts: the student sees the problem, while the teacher additionally receives privileged information such as a reference solution. Token-level distributions are aligned along trajectories sampled from the student policy.

OPSD is a self-distillation variant of OPD rather than merely a smaller OPD run. “Lightweight” here means smaller models and datasets, shorter rollouts, limited training budgets, and focused comparisons of KL/JSD objectives, accuracy, token cost, memory use, and stability.

### Chinese data and post-training plan

Post-training data is pinned to commit `6b952cc50427c84eac543d0b38a8066208433847` of [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/6b952cc50427c84eac543d0b38a8066208433847) on Hugging Face for reproducibility. The repository contains Chinese-heavy and multilingual pretraining, SFT, reasoning-distillation, and preference data.

The intended allocation is:

- `sft_mini_512.jsonl` for the main Chinese SFT stage and stable instruction/chat formatting;
- a cleaned `r1_mix_1024.jsonl` subset as question/reference-solution pairs for OPSD;
- an automatically verifiable math/code subset of `r1_mix_1024.jsonl` for GRPO;
- `pretrain_hq.jsonl` only for optional Chinese continued pretraining, reported separately from post-training;
- `dpo.jsonl` and `rlaif-mini.jsonl` reserved for later preference-optimization work rather than the first OPSD–GRPO comparison.

Before training, the data will be deduplicated, format-checked, filtered for Chinese content, checked for answer verifiability, and split to prevent train/evaluation leakage. Because the repository aggregates multiple upstream sources and lists both Apache-2.0 and CC-BY-NC-2.0, this project will preserve source and commit metadata, follow the stricter non-commercial restriction, and re-check per-file licensing before redistributing derived data.

### Experimental roadmap: SFT + OPSD versus GRPO

The first post-training study uses one fixed **Dense Transformer + AttnRes** pretrained checkpoint. After Chinese SFT, all runs branch from the same SFT checkpoint and use the same training prompts, rollout limits, and generation-token budget wherever possible:

1. **SFT baseline**, with Chinese supervised fine-tuning only;
2. **SFT → OPSD**, using token-level KL/JSD self-distillation on student on-policy trajectories;
3. **SFT → GRPO**, using outcome rewards only on the automatically verifiable math/code subset.

Experiments will compare Chinese instruction following, exact match / pass@1 on verifiable tasks, generated-token cost, peak memory, wall-clock training time, KL/JSD, GRPO reward, and stability. The SFT-only arm measures the real gain from each post-training method, while OPSD and GRPO share the same starting point and an aligned generation budget. Results will be added only after they have been measured.

### Local environment

Local development uses the Conda environment `ai_workspace`, while the import adapters remain compatible with the `cs336_basics` package layout used in Colab.

```bash
conda activate ai_workspace
python -m pytest
```

---

## References / 参考资料

- [Stanford CS336 course](https://stanford-cs336.github.io/)
- [Stanford CS336 Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
- [Moonshot AI: Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- [Attention Residuals paper](https://arxiv.org/abs/2603.15031)
- [DeepSeekMoE paper](https://arxiv.org/abs/2401.06066)
- [DeepSeekMoE official repository](https://github.com/deepseek-ai/DeepSeek-MoE)
- [Self-Distilled Reasoner: On-Policy Self-Distillation](https://arxiv.org/abs/2601.18734)
- [OPSD reference implementation](https://github.com/siyan-zhao/OPSD)

## Status / 状态

This README is temporary and will evolve with the implementation and measured experiments.

本 README 为暂时版本，将随实现进度和真实实验结果持续更新。
