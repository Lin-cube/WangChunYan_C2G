# C2G 参数高尔夫 —— 交付包

> **作者：WangChunYan ｜ 学号：2023105110151**
> 挑战：在 8 张 H100、10 分钟、16MB 的硬约束下训练更好的语言模型，以 FineWeb 验证集的 **BPB**（越低越好）为客观指标。
> 本交付包 = 官方 starter kit + 完整方案文档 + **真实可跑的端到端验证实验** + 实验基建脚本。

**路线**：从官方 Naive Baseline **1.2244 BPB** 出发，走「数据侧 tokenizer 升级 + 容量分配」路线，目标 **< 1.18**（Level 2），stretch **< 1.12**（Level 3）。

---

## ⚡ 30 秒看懂：这个包有什么真实证据

在**没有 GPU、没有网络**的环境下，本包产出了**真实、可复现、可核查**的实测数据（不是计划值，不是编造值）：

| 臂 | vocab | seed/split | val tok/byte | val_loss | **val_bpb** | artifact |
|----|------:|:----------:|-------------:|---------:|------------:|---------:|
| `v512_s1337` | 512 | 1337/1337 | 0.5265 | 3.3816 | **2.5683** | 1,026,658 B |
| `v2048_s1337` | 2048 | 1337/1337 | 0.3332 | 4.5355 | **2.1797** | 1,169,635 B |
| `v512_s42` | 512 | 42/42 | 0.5456 | 4.2865 | **3.3729** | 1,028,208 B |
| `v2048_s42` | 2048 | 42/42 | 0.3368 | 4.5413 | **2.2033** | 1,171,019 B |

**同切分 A/B**：split 1337 上 ΔBPB = **−0.3885**；split 42 上 ΔBPB = **−1.1696**。**两组独立切分方向完全一致**——大词表显著更优。

**核心发现**：词表变大后 `val_loss` **变差**（3.38→4.54），但 `val_bpb` **变好**（2.57→2.18）——因为 `tokens/byte` 从 0.527 降到 0.333。这是 BPB 作为 tokenizer-agnostic 指标的**机制实证**。

> ⚠️ **口径诚实声明**：这些数字来自 **CPU 缩小规模**训练（本地 2.94MB 语料、0.5–0.7M 参数），**与官方榜单（1.08–1.22）不可比**，**不构成 Level 1 的 baseline 复现**。可迁移的是：指标口径、打包链路、以及 tokenizer 效应的方向。榜单成绩仍待 8×H100 + FineWeb 实测。
> 详见 `WangChunYan_C2G_复现验证报告.md`。

---

## 目录结构

```
WangChunYan_C2G/
│
├── README.md                              ← 本文件（总览 + 复现说明）
│
├── ── 必须交付物 ──────────────────────────────────────────────
├── WangChunYan_C2G_方案草案.md             ← 算力申请门槛文档（≥500字，4问）
├── WangChunYan_C2G_方案设计.md             ← 目标Level/技术选型/实验矩阵/16MB预算
├── WangChunYan_C2G_AI日志.md               ← 全过程 AI 协作记录（7 阶段 18+ 轮）
├── WangChunYan_C2G_AAR.md                 ← 复盘报告（含真实失败经验）
│
├── ── 实测证据（真实运行产出）─────────────────────────────────
├── WangChunYan_C2G_复现验证报告.md          ← ★ 真实实测数据报告
├── WangChunYan_C2G_ablation.md            ← 消融实验（实测 + 计划两部分）
├── WangChunYan_C2G_leaderboard.md         ← BPB 对比表
├── WangChunYan_C2G_submission.json        ← 提交元数据（已用实测值回填）
├── WangChunYan_C2G_submission.tar.gz      ← ★ <16MB artifact（1,157,873 B）
├── logs/                                  ← ★ 真实训练日志
│   ├── v512_s1337.log  v2048_s1337.log    ← 主对照（seed 1337）
│   ├── v512_s42.log    v2048_s42.log      ← 第二组种子
│   └── smoke.log                          ← 冒烟测试记录
├── artifacts/                             ← ★ 真实产物
│   ├── *_model.int8.ptz                   ← int8+zlib 压缩模型
│   ├── *_tokenizer.json.zlib              ← 压缩 tokenizer
│   ├── *_result.json                      ← 机器可读实测结果
│   └── *_MANIFEST.json                    ← 打包清单（含 SHA-256）
│
├── ── 可运行代码 ──────────────────────────────────────────────
├── WangChunYan_C2G_verify_cpu.py           ← ★ 真实端到端验证实验（CPU 可跑）
├── WangChunYan_C2G_train_gpt.py            ← 官方基线训练脚本（未改动）
├── WangChunYan_C2G_pack_artifact.py         ← 打包 <16MB artifact
├── analyze_logs.py                        ← 日志/结果分析工具
├── run_experiments.sh                     ← 8×H100 实验矩阵一键运行
├── requirements.txt                       ← 依赖清单
│
├── ── 参考资料 ────────────────────────────────────────────────
├── WangChunYan_C2G_拿来说明.md              ← 拿了什么/改了什么/为什么
├── parameter-golf/                        ← 官方 starter kit（完整，227 文件）
└── reference_baseline/                    ← 官方 baseline 的 submission.json + train.log
```

---

## 快速开始

### 0. 先交方案（Level 1 的门）

`WangChunYan_C2G_方案草案.md` 是**算力申请的前置门槛**（≥500 字、回答 4 个门槛问题）。先交它，审核通过后才发 $25 算力券。

### 1. ★ 跑真实验证实验（本地 CPU，无需 GPU / 无需联网）

```bash
pip install torch numpy          # 仅需这两个依赖

# 冒烟测试（约 40 秒，确认环境 OK）
python WangChunYan_C2G_verify_cpu.py --smoke --corpus-root parameter-golf --tag smoke

# 对照组 A：vocab 512
python WangChunYan_C2G_verify_cpu.py --vocab 512  --steps 250 --seq-len 128 \
  --batch-tokens 2048 --layers 4 --dim 128 --seed 1337 \
  --bpe-sample-bytes 400000 --corpus-root parameter-golf --tag v512_s1337

# 对照组 B：vocab 2048
python WangChunYan_C2G_verify_cpu.py --vocab 2048 --steps 250 --seq-len 128 \
  --batch-tokens 2048 --layers 4 --dim 128 --seed 1337 \
  --bpe-sample-bytes 400000 --corpus-root parameter-golf --tag v2048_s1337

# 汇总成 Markdown 消融表
python analyze_logs.py --results artifacts --markdown

# 打包 <16MB artifact
python WangChunYan_C2G_pack_artifact.py
```

### 2. 远程 8×H100（榜单级，待算力券）

```bash
cd parameter-golf
pip install torch>=2.3 sentencepiece numpy tqdm flash-attn

python3 data/cached_challenge_fineweb.py --variant sp1024   # baseline 数据
python3 data/cached_challenge_fineweb.py --variant sp4096   # Level 2 数据

cd ..
bash run_experiments.sh e001   # 复现官方 baseline 1.2244
bash run_experiments.sh all    # 跑完 3-seed 关键实验
python analyze_logs.py logs/*.log
```

---

## 关键指标口径

| 指标 | 含义 | 取值 |
|------|------|------|
| `val_bpb` | 越低越好，tokenizer-agnostic 压缩率 | baseline 1.2244 → 目标 <1.18 |
| `val_loss` | token 交叉熵（nats） | **辅助**；注意它可能与 BPB 给出相反结论（见实测） |
| `tokens/byte` | 每字节对应多少 token | 词表越大越低 → BPB 越低 |
| `submission_size` | code bytes + int8+zlib 模型 bytes | 必须 ≤ 16,000,000 |
| `train_time` | wallclock | 必须 ≤ 600s（目标 480s 留 buffer） |

---

## 诚实性声明

- `WangChunYan_C2G_submission.json` 顶层的 `val_bpb/val_loss/bytes_total` 是**真实实测值**（不再是模板 0.0），并通过 `metric_scope` 字段显式标注口径为 `cpu_scale_verification_NOT_leaderboard`。
- **榜单级目标值（<1.18 / <1.12）单独存放于 `leaderboard_target` 字段，标注为待实测的计划值**，与实测值绝不混用。
- **官方 baseline 1.2244 的复现仍需 8×H100 + FineWeb**（本环境无网络、无 GPU），本包未声称已完成该复现。
- 训练脚本取自官方 [openai/parameter-golf](https://github.com/openai/parameter-golf)，逻辑未改动；我的增量 = 方案设计 + 真实验证实验 + 实验基建 + 对照/消融 + AI 日志 + AAR。

---

## 学号

**2023105110151**（作者：WangChunYan）
