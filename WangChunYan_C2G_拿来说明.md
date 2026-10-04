# WangChunYan_C2G 拿来说明

> 规则要求：说明**拿了什么、改了什么、为什么改**。这三句话就是工程能力证明。

---

## 1. 拿了什么（来源清单）

| 拿的东西 | 来源 | 许可/依据 |
|----------|------|-----------|
| `train_gpt.py`（基线训练脚本，1126 行） | OpenAI [openai/parameter-golf](https://github.com/openai/parameter-golf) 官方仓库 | MIT 许可，随 starter kit 提供 |
| `data/cached_challenge_fineweb.py` 数据管线 | 同上 | 同上 |
| Muon 优化器（Newton-Schulz 正交化） | 源自 [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)，经官方 train_gpt.py 内建 | THIRD_PARTY_NOTICES.md 已归因 |
| 历史榜单记录（records/） | 官方仓库 `records/track_10min_16mb/` | 规则允许复用历史 top 方案 |
| SentencePiece 词表升级思路 | 榜单多条 SP4096/SP8192 记录 + Kudo & Richardson (EMNLP 2018) | 公开 |

## 2. 改了什么

**核心原则：只加「可解释、可消融」的增量，不重写轮子。**

| 改动 | 改动点 | 为什么改 |
|------|--------|---------|
| ① Tokenizer SP1024 → SP4096/SP8192 | 数据预处理（重 tokenize，不碰模型） | BPB = bits/token × tokens/byte，大词表提高 token 平均字节数 → tokens/byte 下降 → BPB 下降；榜单 SP4096/SP8192 记录单调优于 SP1024 |
| ② `MLP_MULT=2 → 3` | 只改一个 env 变量，模型容量分配 | 榜单多条 "MLP3x / 3x MLP" 记录（如 1.1502、1.1458）证明 relu² MLP 加宽有正收益 |
| ③ `QK_GAIN_INIT=1.5 → 4.5` | 只改一个 env 变量，注意力 logit 增益 | 榜首链 "QK-Gain 5.0/5.25"（1.0828、1.0810）证明 QK-gain 是低风险收益点 |
| ④ 实验基础设施 | 新增 `run_experiments.sh` + `analyze_logs.py` | 把「假设→对照→分析」固化成可复现脚本，避免手工记录误差 |
| ⑤ **真实端到端验证实验** | 新增 `WangChunYan_C2G_verify_cpu.py`（+ `WangChunYan_C2G_pack_artifact.py`） | 在无 GPU/无网络环境下产出**真实实测证据**：忠实移植官方 BPB 口径与 int8+zlib 打包链路，自实现字节级 BPE（不依赖 sentencepiece），并让 BPB 计算**运行时自证**。这是对审计意见「无实测证据」的直接修复 |

## 3. 为什么这么改（三句话）

1. **拿了**官方现代基线（Muon + RMSNorm + RoPE + GQA + U-Net skip + int8 打包），因为它已经是最优起点，从零写只会更差。
2. **改了**「数据侧 tokenizer」和「容量分配」这两个基线**没覆盖**的杠杆——基线把这些都写死成了 `vocab_size=1024 / mlp_mult=2 / qk_gain=1.5` 的默认值。
3. **为什么**：这三个改动每一个都有榜单记录的**独立证据**支撑，且互相正交（tokenizer 改数据、MLP 改容量、QK-gain 改注意力），适合做干净的消融，而不是打包成黑箱「一起上」。

---

## 4. 归属与诚实声明

- `train_gpt.py` 版权归 OpenAI / 原作者，遵循 MIT 与 THIRD_PARTY_NOTICES.md；我未改动其逻辑，仅作为训练脚本交付。`verify_cpu.py` 中忠实移植的官方组件（RMSNorm / Rotary / CausalSelfAttention / MLP / Block / GPT / Muon / eval_val 口径 / quantize_state_dict_int8）同样归因于官方仓库。
- 我的增量贡献 = 实验设计（方案草案/方案设计）+ 真实端到端验证实验（`verify_cpu.py`）+ 实验矩阵脚本 + 日志分析工具 + artifact 打包工具 + 对照/消融 + AI 日志 + AAR。
- 所有 BPB 数字分两类存放，**绝不混用**：
  - **实测值**（真实跑出，CPU 缩小规模）：vocab 512 → `2.5683`，vocab 2048 → `2.1797`，见 `logs/` 与 `artifacts/*_result.json`；口径已在 `submission.json` 的 `metric_scope` 字段标注为 `cpu_scale_verification_NOT_leaderboard`。
  - **计划值**（未实测）：官方 baseline `1.2244` 的复现、Level 2 `<1.18`、Level 3 `<1.12`，见 `submission.json` 的 `leaderboard_target` 字段。
- **未声称**：本包未声称完成官方 baseline 复现，也未声称取得任何榜单成绩。

---

*作者：WangChunYan ｜ 学号：2023105110151*
