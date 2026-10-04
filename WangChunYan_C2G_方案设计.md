# WangChunYan_C2G 方案设计文档

> 在《方案草案》基础上扩展：目标 Level、技术选型、实验矩阵、参数预算、风险与验证方案。
> 作者：WangChunYan ｜ 学号：2023105110151

---

## 1. 目标 Level 与成功标准

| 项 | 内容 |
|----|------|
| **目标 Level** | Level 2（单点改进，BPB < 1.18），并以 Level 3（< 1.12）为 stretch goal |
| **主指标** | `val_bpb`（越低越好，tokenizer-agnostic） |
| **辅助指标** | `val_loss`、`tokens_per_byte`、`train_time`、`submission_size` |
| **统计要求** | 每个配置 ≥ 3 seed，报均值 ± std；新 record 相对旧值需 ≥ 0.005 且 p < 0.01 |
| **硬约束** | wallclock ≤ 600s（目标 480s 留 buffer）；artifact ≤ 16,000,000 bytes |

---

## 2. 技术选型（基线已具备的能力 vs 我要引入的能力）

### 2.1 基线（官方 `train_gpt.py`）已经具备的能力

读代码确认了基线已经「很现代」，**不是从零开始**：

| 组件 | 基线实现 | 代码位置（train_gpt.py） |
|------|---------|------------------------|
| 优化器 | **Muon**（Newton-Schulz 正交化）+ 分参数组 Adam | `Muon` / `zeropower_via_newtonschulz5`（L96-168） |
| 归一化 | RMSNorm（无 bias） | `RMSNorm`（L500） |
| 位置编码 | **RoPE**（rotary） | `Rotary` / `apply_rotary_emb`（L524-552） |
| 注意力 | GQA（8 heads / 4 KV heads）+ Flash SDP | `CausalSelfAttention`（L555） |
| MLP | **relu²**（modded-nanoGPT 风格） | `MLP`（L606） |
| 残差 | **U-Net 式 skip + resid_mix + attn/mlp scale** | `Block`（L620） |
| 激活软截断 | logit softcap（tanh） | `GPT.forward`（L723） |
| 量化 | int8 per-row + zlib 打包 | `quantize_state_dict_int8`（L342） |
| 分布式 | DDP + 8 GPU grad accum | `main`（L731） |

**结论**：官方基线已经把「Modded-nanoGPT 的现代配方」内建好了（Muon、RMSNorm、RoPE、GQA、U-Net skip、softcap、int8 打包）。**我的增量价值不在于重写这些，而在于「数据侧 tokenizer」和「容量分配」这两个基线没覆盖的杠杆。**

### 2.2 我要引入的能力（技术选型）

| 优先级 | 改动 | 类型 | 预期 BPB 收益 | 风险 |
|--------|------|------|--------------|------|
| P0 | **Tokenizer SP1024 → SP4096/SP8192** | 数据侧 | -0.03 ~ -0.05 | 低（榜单多份验证） |
| P1 | **MLP_MULT 2 → 3**（"3x MLP"） | 容量分配 | -0.01 ~ -0.02 | 低（榜单 "MLP3x" 多次出现） |
| P2 | **QK_GAIN_INIT 1.5 → 4~5**（QK-gain） | 注意力增益 | -0.01 ~ -0.02 | 中（需验证收敛） |
| P3 | **更高 weight decay（Muon WD）** | 优化器 | -0.005 ~ -0.01 | 中 |
| 备选 | Parallel residuals / depth recurrence | 架构 | -0.02 ~ -0.05 | 高（Level 3 再上） |

**决策原则**：先做「数据侧 + 容量分配」这类**低风险、高确定性**的改动，逐个消融；架构级改动（parallel residuals、depth recurrence）留到 Level 3 组合阶段。

---

## 3. 实验矩阵（完整版）

### 3.1 阶段一：复现校准（Level 1）

| ID | 配置 | 目的 | 判定标准 |
|----|------|------|---------|
| E001 | baseline（SP1024） | 复现官方 1.2244 | \|实测 - 1.2244\| ≤ 0.005 |

### 3.2 阶段二：单点改进（Level 2）

| ID | 改动 | 命令关键参数 | 预期 BPB | 对照关系 |
|----|------|-------------|----------|---------|
| E002 | SP4096 | `--variant sp4096` + `VOCAB_SIZE=4096` | ≈1.19 | E002 vs E001 |
| E003 | SP8192 | `--variant sp8192` + `VOCAB_SIZE=8192` | ≈1.18 | E003 vs E002（边际） |
| E004 | + MLP3x | `MLP_MULT=3` | ≈1.16~1.17 | E004 vs E002 |

### 3.3 阶段三：组合优化（Level 3 预研）

| ID | 组合 | 预期 BPB | 消融目的 |
|----|------|----------|---------|
| E005 | SP4096 + MLP3x + QK-gain | ≈1.15~1.16 | QK-gain 增量 |
| E006 | E005 + Muon WD 上调 | ≈1.14~1.15 | WD 增量 |
| E007 | E006 + 量化（embedding int6） | ≈1.13~1.14 | 量化 + 省空间多训 |

**每个实验跑 3 seed**：SEED ∈ {1337, 42, 2025}，最终报均值 ± std。

---

## 4. 参数预算与 16MB 约束分析

### 4.1 为什么 tokenizer 升级是「白赚」的

在 tied embedding 下，embedding 参数量 = `vocab_size × model_dim`：

| vocab | embedding 参数（512 维） | fp32 体积 | 说明 |
|-------|------------------------|----------|------|
| 1024 | 524,288 | ~2.1MB | baseline |
| 4096 | 2,097,152 | ~8.4MB | 4× |
| 8192 | 4,194,304 | ~16.8MB | 8×（原始 fp32 已超） |

**关键洞察**：虽然大词表 embedding 参数翻倍，但（1）训练用 bf16、导出用 int8+zlib，实际体积远小于 fp32；（2）zlib 对「低频 token 的稀疏 embedding 行」压缩率极高（大量接近零的行）；（3）官方 `quantize_state_dict_int8` 是 per-row 量化，对 embedding 天然友好。**所以 SP8192 在 16MB 内是可行的**，榜单榜首正是 SP8192 组合。

### 4.2 16MB 预算分配（目标）

| 组成 | baseline 实测 | 我的目标 |
|------|--------------|---------|
| code bytes | 47,642 | ≈ 48,000（改动极小） |
| model int8+zlib | 15,815,847 | < 15,900,000 |
| **合计** | **15,863,489** | **< 15,950,000（留 ≥ 50KB buffer）** |

---

## 5. 数据准备方案

```bash
# 官方缓存数据 + tokenizer 下载（仅一次）
python3 data/cached_challenge_fineweb.py --variant sp1024   # baseline
python3 data/cached_challenge_fineweb.py --variant sp4096   # Level 2
python3 data/cached_challenge_fineweb.py --variant sp8192   # Level 2 对比
```

> 评估集 `fineweb_val_*` 是官方固定的「前 50k 文档」，**不可下载评估集、不可联网**。训练数据只用于训练，验证集只用于最终 eval，**绝不用验证集调超参**（用 train loss 或 held-out，避免过拟合验证集——这是官方「血的教训」清单第 4 条）。

---

## 6. 训练命令模板

```bash
# 阶段一 baseline（复现校准）
NCCL_IB_DISABLE=1 RUN_ID=e001_baseline \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# 阶段二 SP4096 单点改进
NCCL_IB_DISABLE=1 RUN_ID=e002_sp4096 \
  DATA_PATH=./data/datasets/fineweb10B_sp4096 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
  VOCAB_SIZE=4096 SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# 阶段三 组合（SP4096 + MLP3x + QK-gain）
NCCL_IB_DISABLE=1 RUN_ID=e005_combo \
  DATA_PATH=./data/datasets/fineweb10B_sp4096 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
  VOCAB_SIZE=4096 MLP_MULT=3 QK_GAIN_INIT=4.5 SEED=1337 \
  MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## 7. 风险清单与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 大词表 embedding 超 16MB | 中 | 提交被拒 | per-row int8 + zlib 通常足够；不够则 embedding int6 / GPTQ |
| SP8192 训练不收敛 | 低 | BPB 变差 | 回退 SP4096；重调 `TIED_EMBED_LR` |
| 10 分钟踩边 | 中 | 被官方环境判超时 | 目标 480s，预留 20% buffer |
| 过拟合验证集 | 中 | 官方评估差很多 | 只用 train loss 调超参，验证集仅最终 eval |
| 单 seed 波动 | 高 | BPB ±0.01 | 3 seed 取均值，报 ±std |

---

## 8. 与 rubric 的映射

| rubric 维度 | 我的交付 |
|-------------|---------|
| 成绩达成（25） | BPB 对比表 + 3 seed 记录 + 边际提升策略 |
| 方法学（20） | 假设→对照→分析闭环 + 消融表 + 可解释策略 |
| 产物完整性（15） | 完整脚本 + README + 安装/使用说明 + submission.json |
| AI 使用（20） | AI 日志（多轮迭代、prompt 优化、工作流设计） |
| 复盘（20） | AAR（具体问题分析 + 改进方案 + 失败经验） |

---

*作者：WangChunYan ｜ 学号：2023105110151*
