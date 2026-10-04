# WangChunYan_C2G 榜单对比（BPB）

> 我的方案 vs 官方 baseline vs 当前 SOTA，数据来自官方 `records/` 榜单。

## 1. 关键锚点

| 方案 | val_bpb | 说明 |
|------|---------|------|
| **当前 SOTA**（SP8192 + 3-Layer Recurrence + ParResid + Legal TTT） | **1.0810** | bigbag, 2026-04-09 |
| SP8192 + QK-Gain 5 + Legal TTT | 1.0828 | dexhunter |
| SP8192 + GPTQ Embeddings + Depth Recurrence | 1.0856 | Kevin Clark |
| SP4096 + Depth Recurrence + ParResid + MuonEq-R | 1.0897 | aryanbhosale |
| 4096-Vocab + Larger Model + High WD | 1.0979 | Kevin Clark |
| **Level 3 门槛** | **< 1.12** | 目标 |
| **Level 2 门槛** | **< 1.18** | 目标 |
| **Naive Baseline** | **1.2244** | 官方基线 |
| 我的目标（SP4096 + MLP3x + QK-gain） | **≈1.15~1.18** | 本方案 |

## 2. 官方榜单演进（压缩版，说明我的路线依据）

| 阶段 | 代表记录 | BPB | 关键改动 |
|------|---------|-----|---------|
| 起点 | Naive Baseline | 1.2244 | 9x512 KV4 SP1024 |
| 早期单点 | fp16 Embed / int6 mixed | 1.2147~1.2197 | 精度/量化 |
| 上下文 | 2048/4096 seq | 1.2014~1.206 | 序列长度 |
| 量化深入 | Int6 MLP3x + SmearGate | 1.1458~1.1502 | **MLP3x 出现** |
| 架构 | 11L XSA + EMA | 1.1248~1.1307 | 深度/XSA |
| tokenizer 升级 | 4096-Vocab / SP4096 | 1.0897~1.0979 | **SP4096 出现** |
| 顶层组合 | SP8192 + Recurrence + TTT | 1.0810~1.0856 | **SP8192 + 架构** |

**读法**：BPB 从 1.2244 到 1.08 的路径上，**tokenizer 升级（SP4096/8192）是「1.1x → 1.08」段的关键跳变**，这正好是我第一步选的方向；MLP3x 和 QK-gain 是「1.15 → 1.12」段的稳定贡献项，是我第二步、第三步。

## 3. 我的定位

```
                    BPB
1.25 ┤
1.2244 ┤ ● Naive Baseline（起点）
1.20  ┤
1.18  ┤ ─ ─ Level 2 门槛 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
1.16  ┤ ● 我的目标（SP4096 + MLP3x + QK-gain）
1.14  ┤
1.12  ┤ ─ ─ Level 3 门槛 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
1.10  ┤ ● 4096-Vocab + High WD (1.0979)
1.08  ┤ ● SOTA (1.0810)
```

---

## 4. ★ 真实实测锚点（CPU 规模验证，**与上表不可比**）

上面 §1–§3 是**官方榜单数据 + 我的计划目标**。下表是**本包真实跑出来的实测值**，口径与官方一致但规模不同，**列入此处仅为留痕，不参与榜单对比**：

| 实测臂 | vocab | val tok/byte | val_loss | **val_bpb** | artifact(bytes) | 口径 |
|--------|------:|-------------:|---------:|------------:|----------------:|------|
| `v512_s1337` | 512 | 0.5265 | 3.3816 | **2.5683** | 1,026,658 | CPU 缩小规模 |
| `v2048_s1337` | 2048 | 0.3332 | 4.5355 | **2.1797** | 1,169,635 | CPU 缩小规模 |

**为什么不与榜单并列**：语料是本地 2.94MB 文本（非 FineWeb）、硬件是 CPU（非 8×H100）、模型 0.5–0.7M 参数（官方约 17M）、tokenizer 是自实现字节级 BPE（非官方 SentencePiece）。**绝对数值没有可比性**；有意义的是**相对变化**：同一预算下 vocab 512→2048 使 BPB **下降 0.3886**，且 artifact 仅增 143 KB。

**关键机制**（实测）：`val_loss` 从 3.38 **升到** 4.54，但 `val_bpb` 从 2.57 **降到** 2.18——因为 tokens/byte 从 0.527 降到 0.333。这实证了 BPB 作为 tokenizer-agnostic 指标的必要性。

**当前状态**：
- ✅ 有真实实测证据（日志 / artifact / 机器可读结果 JSON）
- ⏳ 官方 baseline 1.2244 的复现与 Level 2/3 目标成绩，**仍待 8×H100 + FineWeb**（本环境无 GPU、无网络）

详见 `WangChunYan_C2G_复现验证报告.md`。

---

*作者：WangChunYan ｜ 学号：2023105110151*
