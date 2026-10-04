# WangChunYan_C2G 方案草案（算力申请用）

> 挑战：C2G 参数高尔夫 —— 极限约束下的语言模型训练
> 目标基线：Naive Baseline = **1.2244 BPB**（9 层 / 512 维 / tied embedding / 4 KV head）
> 申请等级：Level 1（$25 算力券）+ Level 2 单点改进路线预研
> 作者：WangChunYan ｜ 学号：2023105110151

---

## 0. 一句话结论

我把第一步的改进方向锁定在 **Tokenizer 升级（SentencePiece 1024 → SP4096/SP8192）**。这是一个「改动集中在数据预处理、不碰模型主体、历史收益明确、可被多份榜单记录交叉验证」的方向，预计把 BPB 从 1.2244 压到 **1.18 以下**（Level 2 门槛），并为后续组合优化（架构 + 量化 + 优化器）释放参数预算。

---

## 1. 门槛问题一：打算把 baseline 往哪个方向压？

**选一个方向：Tokenizer 升级（vocab size 1024 → 4096/8192）。**

理由（为什么不是「都试试」）：

- 在 16MB 硬约束下，小模型的参数里 **embedding 占了极大的比例**。baseline 用 tied embedding，`vocab_size=1024, model_dim=512`，输入/输出 embedding 只有 `1024 × 512 = 524,288` 个参数（约 1MB fp32 / 0.5MB bf16），这被官方代码注释明确点名为「小模型 embedding 体积巨大」的关键瓶颈。
- 增大 vocab 的收益是**双重**的：一是**每个 token 平均携带更多原始字节**（BPB = bits-per-token × tokens-per-byte，分母变大、分子不变，BPB 直接下降）；二是**压缩了序列长度**，同样的字节数对应更短的 token 序列，模型在同等计算量下能「看到」更长的上下文。
- 这个方向**不改变模型主体结构**，因此是风险最低、可解释性最强、最容易写出「假设 → 对照 → 分析」闭环的第一个改动。

目标（可量化）：

| 指标 | Baseline | 目标 |
|------|----------|------|
| val_bpb | 1.2244 | **< 1.18**（Level 2） |
| 压缩 artifact | 15,863,489 bytes | < 16,000,000 bytes（留 buffer） |
| wallclock | ~600s | ≤ 600s（预留 20% → 目标 480s） |

---

## 2. 门槛问题二：怎么知道这个方向有效？（证据）

我读了官方 `records/` 里的公开榜单与训练日志，**证据不是来自猜，而是来自历史提交的交叉验证**：

1. **官方榜单记录（leaderboard 直接证据）**：当前 SOTA 与多个前几名方案都把「SP4096 / SP8192」写进了命名与摘要，例如：
   - `SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT` → **1.0810**（榜首）
   - `SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R` → **1.0897**
   - `4096-Vocab + Larger Model + High WD + Simplifications` → **1.0979**（摘要明确写 "SP4096"）
   - 对比没有 tokenizer 升级的早期记录（`2048 seq length` = 1.206、`int6 mixed precision` = 1.2147、`Naive Baseline` = 1.2244），**vocab 从 1024 提升到 4096/8192 是榜单从 1.2x 区间进入 1.1x 区间的关键转折点之一**。

2. **BPB 指标的定义（理论证据）**：官方 `train_gpt.py` 的 `eval_val` 用 `bits_per_token = val_loss / ln2`、`tokens_per_byte = token_count / byte_count`、`val_bpb = bits_per_token × tokens_per_byte`。BPB 是 tokenizer-agnostic 的：**tokenizer 越好，token 平均字节数越高，`tokens_per_byte` 越低，在 val_loss 不变甚至略升的情况下 BPB 仍会下降**。这是 `train_gpt.py` 第 171-177 行的代码事实，不是猜测。

3. **论文级依据**：`SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing`（Kudo & Richardson, EMNLP 2018）说明 BPE/unigram 类子词模型通过增大词表能降低序列长度、提升编码效率；这与榜单「SP8192 优于 SP4096 优于 SP1024」的单调趋势一致。

**结论**：这个方向有「代码事实 + 榜单多份记录 + 论文」三重证据支撑，属于高置信度、高 ROI 的第一步。

---

## 3. 门槛问题三：跑多少次实验？每次看什么指标？$25 怎么花？

### 实验矩阵（Level 1 → Level 2 预研，共 5 次完整 8×H100 跑 + 本地小尺寸验证）

| 实验 | 内容 | 预期 BPB | 作用 |
|------|------|----------|------|
| E001 | Baseline 复现（SP1024） | ≈ 1.2244 | 校准：验证我的环境与官方一致（±0.005） |
| E002 | SP4096 tokenizer | ≈ 1.19 | 单点改进，验证 vocab 收益 |
| E003 | SP8192 tokenizer | ≈ 1.18 | 对比 4096 vs 8192 的边际收益 |
| E004 | SP4096 + MLP_MULT=3 | ≈ 1.16~1.17 | 组合第一步：更大的 MLP |
| E005 | E004 + QK_GAIN_INIT 上调 | ≈ 1.15~1.16 | 组合第二步：QK-gain |

每个实验跑 **3 个 seed**（1337 / 42 / 2025），报均值 ± std。

### 每次要看的指标（不是我瞎看，是每行都有目的）

- `val_bpb`（核心，越低越好）+ `val_loss`（辅助，区分是 tokenizer 收益还是真实建模收益）
- `tokens_per_byte`（验证 tokenizer 是否真的提升了编码效率）
- `train_time / step_avg`（确认 ≤ 600s wallclock，预留 buffer）
- `Total submission size int8+zlib`（确认 ≤ 16,000,000 bytes）
- `train_loss` 曲线（判断是否收敛、是否发散）

### $25 怎么花（省钱是第一原则）

| 用途 | 成本 | 说明 |
|------|------|------|
| 本地/单卡 A100 小尺寸验证 | ~$0（先用自己的 GPU / 学校集群） | 用 `TRAIN_SEQ_LEN=512`、`ITERATIONS=500` 跑 smoke test，验证代码无 bug |
| E001 baseline 复现 | ~$2（1 次 8×H100） | 校准环境 |
| E002–E003 tokenizer 对照 | ~$4（2 次） | 单点改进 |
| E004–E005 组合预研 | ~$4（2 次） | Level 2 预研 |
| **小计** | **~$10** | **预留 $15 用于 seed 重复 + 意外重跑** |

> 核心思路：**80% 的实验在便宜硬件上验证思路，只在 8×H100 上跑最后的关键对照**。$25 可以覆盖 5 次 8×H100 + 大量本地 smoke test，绰绰有余。

---

## 4. 门槛问题四：失败怎么办？

我把「失败」拆成三类，每类都有明确的下一步（这一步最能说明我是否真的想清楚了）：

**失败类型 A：BPB 没降反而升了（例如 SP8192 反而比 SP4096 差）。**
→ 下一步：先查 `tokens_per_byte` 是否真的提高了（如果是，说明 BPB 变差来自 `val_loss` 变差，即模型容量不够消化大词表）；然后**回退到 SP4096**，把省下的 embedding 参数预算转到 `MLP_MULT` 或 `NUM_LAYERS`，重新跑对照。**结论永远是「哪个 vocab 在固定 16MB 下 BPB 最低」，而不是「词表越大越好」。**

**失败类型 B：训练不收敛 / loss 发散。**
→ 下一步：先跑小尺寸 smoke test 定位是 tokenizer 数据问题还是学习率问题；检查 `TIED_EMBED_LR`（大词表 embedding 需要重调 LR）；必要时用 `GRAD_CLIP_NORM`。**不猜，二分定位。**

**失败类型 C：artifact 超 16MB。**
→ 下一步：baseline 已经用 int8 + zlib 打包（官方代码 `quantize_state_dict_int8`），tokenizer 升级会增大 embedding，但 zlib 对大词表的稀疏 embedding 压缩率通常更高；若仍超，启用 `CONTROL_TENSOR_NAME_PATTERNS` / 更激进的 embedding 量化。**训练循环末尾自动检查 artifact 大小，超了立即知道，不留到提交。**

**兜底承诺**：如果 tokenizer 方向整体证伪（BPB 无显著下降），我会在 AAR 里如实记录「为何这个方向不成立」，并把结论写成**可复用的负结果**——这本身也是有效的方法学交付，而不是硬凑一个「看起来有效」的假结果。

---

## 5. 交付承诺（对应 rubric）

- ✅ 可复现的训练脚本 + 3 seed 日志（**复现性**）
- ✅ 假设 → 对照 → 分析的完整闭环（**方法学**）
- ✅ 每个组件的消融 + BPB 对比表（**成绩达成 + 方法学**）
- ✅ 全程 AI 协作日志 + AAR 复盘（**AI 使用 + 复盘**）

---

*学号：2023105110151 ｜ 提交日期：以最终提交为准*
