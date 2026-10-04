#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WangChunYan_C2G_verify_cpu.py
================================================================================
C2G 参数高尔夫 —— CPU 规模端到端复现验证 + Tokenizer 消融实验

【这个脚本解决什么问题】
审计指出本交付包「没有任何实测证据」：logs/ 为空、submission.json 指标全 0。
本脚本在**没有 GPU、没有网络**的受限环境下，跑出一组**真实、可复现、可核查**的
实验数据，用来证明：

  1. 官方 BPB 指标的计算链路是对的（bits/token x tokens/byte，tokenizer-agnostic）
  2. 官方 int8+zlib 量化打包链路真的能把模型压到目标体积，且往返精度可测
  3. 本方案的核心假设「更大词表 -> tokens/byte 下降 -> BPB 下降」**真实成立**（实测）

【与官方 train_gpt.py 的关系（拿来主义）】
本文件忠实移植了官方 train_gpt.py 的以下组件，逻辑一一对应：
  - RMSNorm / Rotary / apply_rotary_emb / CausalSelfAttention(GQA) / MLP(relu^2)
  - Block（resid_mix + attn_scale + mlp_scale）
  - GPT（U-Net skip + tied embedding + logit softcap）
  - Muon 优化器（Newton-Schulz 正交化 + nesterov + scale correction）
  - eval_val 的 BPB 口径（bits_per_token * tokens_per_byte）
  - quantize_state_dict_int8（per-row int8 + zlib）
差异仅在于：把 CUDA/DDP/FlashAttention 换成 CPU 可跑的等价实现，并把规模缩小到
单机 CPU 可承受的范围。**规模缩小不等于方法不同**——指标口径与打包链路完全一致。

【语料】
使用本地已有文本（官方 repo 的 .md/.py/.txt，共约 2.9MB），按文件切分 train/val，
避免验证集泄漏。真实 FineWeb 需要联网下载，本机网络不可达，故用本地语料替代，
并在报告中明确标注这一限制。

【用法】
  python WangChunYan_C2G_verify_cpu.py --vocab 512  --steps 200 --tag v512
  python WangChunYan_C2G_verify_cpu.py --vocab 2048 --steps 200 --tag v2048
  python WangChunYan_C2G_verify_cpu.py --smoke          # 极速冒烟测试
================================================================================
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ==============================================================================
# 0. 配置
# ==============================================================================


class Config:
    """实验配置。默认值面向「CPU 上几分钟内跑完」，可通过 CLI 覆盖。"""

    def __init__(self, **kw):
        # ---- 数据 ----
        self.corpus_globs = kw.get("corpus_globs", ["**/*.md", "**/*.py", "**/*.txt"])
        self.corpus_root = kw.get("corpus_root", ".")
        self.val_ratio = kw.get("val_ratio", 0.1)
        self.seed = kw.get("seed", 1337)
        # 数据切分种子。默认 None => 沿用 self.seed（保持与既有日志完全一致的行为）。
        # 单独暴露它是为了支持「纯粹种子复现」：固定切分、只变训练随机性。
        # 教训：最初 seed 同时决定切分与训练随机性，导致跨 seed 比较混杂了两种方差来源。
        self.split_seed = kw.get("split_seed", None)

        # ---- tokenizer ----
        self.vocab = kw.get("vocab", 512)          # 总词表 = 256 字节 token + merges
        self.bpe_merges = max(self.vocab - 256, 0)
        # BPE 训练语料上限（字节）。在 train 集上取样学习 merges 是标准做法，
        # 也让 CPU 上训练大词表成为可能；两个对照组必须用**同一样本**才算公平。
        self.bpe_sample_bytes = kw.get("bpe_sample_bytes", 800_000)

        # ---- 模型形状（对照官方 9L/512d，按比例缩小）----
        self.num_layers = kw.get("num_layers", 4)
        self.model_dim = kw.get("model_dim", 128)
        self.num_heads = kw.get("num_heads", 4)
        self.num_kv_heads = kw.get("num_kv_heads", 2)
        self.mlp_mult = kw.get("mlp_mult", 2)
        self.tie_embeddings = kw.get("tie_embeddings", True)
        self.rope_base = kw.get("rope_base", 10000.0)
        self.logit_softcap = kw.get("logit_softcap", 30.0)
        self.qk_gain_init = kw.get("qk_gain_init", 1.5)
        self.tied_embed_init_std = kw.get("tied_embed_init_std", 0.005)

        # ---- 训练 ----
        self.seq_len = kw.get("seq_len", 128)
        self.batch_tokens = kw.get("batch_tokens", 4096)
        self.steps = kw.get("steps", 200)
        self.warmup_steps = kw.get("warmup_steps", 10)
        self.warmdown_frac = kw.get("warmdown_frac", 0.15)
        self.val_every = kw.get("val_every", 50)
        self.log_every = kw.get("log_every", 25)

        # ---- 优化器 ----
        self.embed_lr = kw.get("embed_lr", 0.05)
        self.matrix_lr = kw.get("matrix_lr", 0.02)
        self.scalar_lr = kw.get("scalar_lr", 0.02)
        self.muon_momentum = kw.get("muon_momentum", 0.95)
        self.muon_backend_steps = kw.get("muon_backend_steps", 5)
        self.beta1 = kw.get("beta1", 0.9)
        self.beta2 = kw.get("beta2", 0.95)
        self.adam_eps = kw.get("adam_eps", 1e-8)

        # ---- 输出 ----
        self.tag = kw.get("tag", "run")
        self.log_dir = Path(kw.get("log_dir", "logs"))
        self.artifact_dir = Path(kw.get("artifact_dir", "artifacts"))


# ==============================================================================
# 1. 语料：从本地文本构建 train/val
# ==============================================================================


def build_corpus(cfg: Config) -> tuple[str, str, dict]:
    """按文件切分语料，避免同一文件同时出现在 train 和 val（防泄漏）。"""
    root = Path(cfg.corpus_root)
    files: list[Path] = []
    for g in cfg.corpus_globs:
        files.extend(root.glob(g))
    # 只看文本类文件，且跳过过小的文件
    files = sorted({f for f in files if f.is_file() and f.stat().st_size > 200})
    if not files:
        raise RuntimeError(f"在 {root} 下未找到语料文件（globs={cfg.corpus_globs}）")

    rng = random.Random(cfg.split_seed if cfg.split_seed is not None else cfg.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * cfg.val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:]

    def read(fs: list[Path]) -> str:
        parts = []
        for f in fs:
            try:
                parts.append(f.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
        return "\n\n".join(parts)

    train_text, val_text = read(train_files), read(val_files)
    meta = {
        "num_files_total": len(files),
        "num_files_train": len(train_files),
        "num_files_val": len(val_files),
        "train_bytes_utf8": len(train_text.encode("utf-8")),
        "val_bytes_utf8": len(val_text.encode("utf-8")),
        "val_file_names": [f.name for f in val_files][:20],
    }
    return train_text, val_text, meta


# ==============================================================================
# 2. 字节级 BPE tokenizer（纯 Python，无外部依赖）
# ==============================================================================
# 设计说明：
#   - 初始字母表 = 256 个字节 token，因此**永不产生 UNK**，且任意 UTF-8 文本都能编码。
#   - 每个 token 都能精确还原为一段字节序列 => token 的字节长度是精确已知的。
#   - 因此 concatenate(所有 token 的字节) == 原始文本字节，tokens/byte 可被**精确验证**。
#     这正是官方 BPB 口径（tokenizer-agnostic）所要求的性质。

# 预分词正则：必须**无损覆盖全部字节**，否则 token 字节数 != 原文字节数，BPB 就错了。
#   - `\s*\S+` 让一段空白跟随其后的词（对应 SentencePiece 的 ▁ 前缀语义）
#   - `|\s+` 兜底纯空白段（如连续 `\n\n`），保证没有任何字节被丢弃
_WORD_RE = re.compile(rb"\s*\S+|\s+")


def _pretokenize(data: bytes) -> Counter:
    """把字节流切成「词块」并统计频率。"""
    chunks = _WORD_RE.findall(data)
    # 自证：切分必须无损，拼接回去必须与原文逐字节相同
    assert b"".join(chunks) == data, "预分词不是无损的，BPB 会算错"
    return Counter(chunks)


def train_bpe(data: bytes, num_merges: int, verbose: bool = False) -> dict:
    """训练字节级 BPE。返回 vocab：{token_id: bytes} 以及 merges 列表。"""
    word_freq = _pretokenize(data)
    # 每个唯一词 -> tuple[bytes]，逐字节开始
    words: dict[tuple, int] = {}
    for w, c in word_freq.items():
        words[tuple(bytes([b]) for b in w)] = c

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    for m in range(num_merges):
        pair_counts: Counter = Counter()
        for symbols, c in words.items():
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += c
        if not pair_counts:
            break
        best, freq = pair_counts.most_common(1)[0]
        if freq < 2:
            break
        merges.append(best)
        merged_token = best[0] + best[1]
        vocab[len(vocab)] = merged_token

        # 应用该 merge 到所有词
        new_words: dict[tuple, int] = {}
        for symbols, c in words.items():
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                    out.append(merged_token)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            key = tuple(out)
            new_words[key] = new_words.get(key, 0) + c
        words = new_words
        if verbose and (m + 1) % 200 == 0:
            print(f"  [bpe] merge {m + 1}/{num_merges} best_freq={freq}", flush=True)

    return {"vocab": vocab, "merges": merges}


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges
        # merge 优先级：越早的 merge 优先级越高
        self.merge_rank = {p: i for i, p in enumerate(merges)}
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)

    def _encode_word(self, word: bytes) -> list[bytes]:
        symbols = [bytes([b]) for b in word]
        while len(symbols) > 1:
            best_rank, best_i = None, None
            for i in range(len(symbols) - 1):
                r = self.merge_rank.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            symbols = (
                symbols[:best_i]
                + [symbols[best_i] + symbols[best_i + 1]]
                + symbols[best_i + 2 :]
            )
        return symbols

    def encode(self, text: str | bytes) -> list[int]:
        data = text.encode("utf-8") if isinstance(text, str) else text
        ids: list[int] = []
        for w in _WORD_RE.findall(data):
            for sym in self._encode_word(w):
                tid = self.token_to_id.get(sym)
                if tid is None:  # 理论上不会发生：字节 token 覆盖全部
                    ids.extend(sym[i] for i in range(len(sym)))
                else:
                    ids.append(tid)
        return ids

    def byte_lengths(self) -> np.ndarray:
        """每个 token 的字节长度 LUT —— BPB 计算的核心（与官方 base_bytes_lut 对应）。"""
        lut = np.zeros(self.vocab_size, dtype=np.int32)
        for tid, b in self.vocab.items():
            lut[tid] = len(b)
        return lut

    def to_jsonable(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
        }

    @staticmethod
    def from_jsonable(obj: dict) -> "BPETokenizer":
        vocab = {int(k): bytes.fromhex(v) for k, v in obj["vocab"].items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in obj["merges"]]
        return BPETokenizer(vocab, merges)


# ==============================================================================
# 3. 模型（忠实移植官方 train_gpt.py 的架构组件）
# ==============================================================================


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        return freqs.cos()[None, None, :, :].to(dtype), freqs.sin()[None, None, :, :].to(dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim 必须为偶数以支持 RoPE"
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), float(qk_gain_init)))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        q = self.c_q(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(s, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        try:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        except TypeError:  # 老版本 torch 无 enable_gqa：手工扩 KV
            rep = self.num_heads // self.num_kv_heads
            y = F.scaled_dot_product_attention(
                q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                attn_mask=None, is_causal=True,
            )
        return self.proj(y.transpose(1, 2).contiguous().reshape(b, s, d))


class MLP(nn.Module):
    """relu^2 MLP（官方 modded-nanogpt 风格）"""

    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tie_embeddings = cfg.tie_embeddings
        self.logit_softcap = cfg.logit_softcap
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.model_dim)
        self.num_encoder_layers = cfg.num_layers // 2
        self.num_decoder_layers = cfg.num_layers - self.num_encoder_layers
        n_skip = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(n_skip, cfg.model_dim))
        self.blocks = nn.ModuleList([
            Block(cfg.model_dim, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_mult,
                  cfg.rope_base, cfg.qk_gain_init)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.model_dim, cfg.vocab, bias=False)
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.cfg.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.cfg.model_dim,))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


# ==============================================================================
# 4. Muon 优化器（移植自官方 train_gpt.py / modded-nanogpt）
# ==============================================================================


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, backend_steps=5, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g.to(dtype=p.dtype).reshape(p.shape), alpha=-lr)


# ==============================================================================
# 5. BPB 评估（忠实移植官方 eval_val 的口径）
# ==============================================================================


def eval_bpb(model: GPT, tokens: Tensor, cfg: Config, byte_lut: np.ndarray,
             batch_seqs: int = 32) -> tuple[float, float, float, int]:
    """返回 (val_loss, val_bpb, tokens_per_byte, 目标 token 数)。

    口径与官方完全一致：
        bits_per_token = val_loss / ln(2)
        val_bpb        = bits_per_token * (token_count / byte_count)
    因为语料按整文件切分、拼接后再切窗口，每个目标 token 的字节数由 LUT 精确给出。
    """
    model.eval()
    total_seqs = (tokens.numel() - 1) // cfg.seq_len
    loss_sum = 0.0
    tok_count = 0
    byte_count = 0
    with torch.inference_mode():
        for start in range(0, total_seqs, batch_seqs):
            end = min(start + batch_seqs, total_seqs)
            raw_s, raw_e = start * cfg.seq_len, end * cfg.seq_len + 1
            local = tokens[raw_s:raw_e].to(torch.int64)
            x = local[:-1].reshape(-1, cfg.seq_len)
            y = local[1:].reshape(-1, cfg.seq_len)
            loss = model(x, y).detach()
            n = y.numel()
            loss_sum += float(loss) * n
            tok_count += n
            byte_count += int(byte_lut[y.reshape(-1).numpy()].sum())
    model.train()
    val_loss = loss_sum / max(tok_count, 1)
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = tok_count / max(byte_count, 1)
    return val_loss, bits_per_token * tokens_per_byte, tokens_per_byte, tok_count


# ==============================================================================
# 6. int8 + zlib 量化打包（忠实移植官方 quantize_state_dict_int8）
# ==============================================================================

KEEP_FLOAT_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
KEEP_FLOAT_MAX_NUMEL = 65_536
CLIP_Q = 99.99984 / 100.0


def _quantize_float_tensor(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), CLIP_Q, dim=1)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)
    return q.contiguous(), scale


def quantize_state_dict_int8(state_dict: dict) -> tuple[dict, dict]:
    quantized, scales, dtypes, passthrough, passthrough_orig = {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "baseline_tensor_bytes",
                           "int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(t.numel()) * int(t.element_size())
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += int(t.numel()) * int(t.element_size())
            continue
        if t.numel() <= KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in KEEP_FLOAT_PATTERNS):
                kept = t.float().contiguous()
            else:
                passthrough_orig[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.numel()) * int(kept.element_size())
            continue
        q, s = _quantize_float_tensor(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += int(q.numel()) * int(q.element_size()) + int(s.numel()) * int(s.element_size())
    obj = {"__quant_format__": "int8_per_row_v1", "quantized": quantized, "scales": scales,
           "dtypes": dtypes, "passthrough": passthrough,
           "passthrough_orig_dtypes": passthrough_orig}
    return obj, stats


def dequantize_state_dict_int8(obj: dict) -> dict:
    out = {}
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if s.ndim > 0:
            out[name] = (q.float() * s.to(torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        od = obj["passthrough_orig_dtypes"].get(name)
        out[name] = t.to(getattr(torch, od)) if isinstance(od, str) else t
    return out


# ==============================================================================
# 7. 训练主流程
# ==============================================================================


def lr_scale(step: int, cfg: Config) -> float:
    """warmup + warmdown（线性）。"""
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    warmdown_steps = max(int(cfg.steps * cfg.warmdown_frac), 1)
    warmdown_start = cfg.steps - warmdown_steps
    if step >= warmdown_start:
        return max((cfg.steps - step) / warmdown_steps, 0.0)
    return 1.0


class TokenStream:
    """顺序消费 token 流，用尽后从头循环（与官方 TokenStream 行为一致）。

    注意：take(n) 返回**恰好 n 个** token；调用方自行多要 1 个用来构造 (x, y) 的错位。
    """

    def __init__(self, tokens: Tensor):
        self.tokens = tokens
        self.pos = 0

    def take(self, n: int) -> Tensor:
        if self.pos + n > self.tokens.numel():
            self.pos = 0  # 循环使用
        out = self.tokens[self.pos : self.pos + n]
        self.pos += n
        return out


def make_batch(stream: TokenStream, cfg: Config) -> tuple[Tensor, Tensor]:
    n_seqs = cfg.batch_tokens // cfg.seq_len
    span = n_seqs * cfg.seq_len + 1
    local = stream.take(span).to(torch.int64)
    x = local[:-1].reshape(n_seqs, cfg.seq_len)
    y = local[1:].reshape(n_seqs, cfg.seq_len)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="C2G CPU-scale verification & tokenizer ablation")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--mlp-mult", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="数据切分种子（默认= --seed，保持既有行为）；固定它可做纯种子复现")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--bpe-sample-bytes", type=int, default=800_000)
    ap.add_argument("--smoke", action="store_true", help="极速冒烟测试")
    ap.add_argument("--corpus-root", type=str, default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cfg = Config(
        vocab=args.vocab, steps=args.steps, seq_len=args.seq_len,
        batch_tokens=args.batch_tokens, num_layers=args.layers, model_dim=args.dim,
        mlp_mult=args.mlp_mult, seed=args.seed, split_seed=args.split_seed,
        bpe_sample_bytes=args.bpe_sample_bytes,
        tag=args.tag or f"v{args.vocab}_s{args.seed}",
    )
    if args.corpus_root:
        cfg.corpus_root = args.corpus_root
    if args.val_every is not None:
        cfg.val_every = args.val_every
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.smoke:
        cfg.steps, cfg.seq_len, cfg.batch_tokens = 20, 64, 1024
        cfg.num_layers, cfg.model_dim, cfg.val_every, cfg.log_every = 2, 64, 10, 5
        cfg.vocab = min(cfg.vocab, 320)
        cfg.bpe_merges = max(cfg.vocab - 256, 0)

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.log_dir / f"{cfg.tag}.log"

    def log0(msg: str, console: bool = True) -> None:
        if console:
            print(msg, flush=True)
        with open(logfile, "a", encoding="utf-8") as f:
            print(msg, file=f)

    # 幂等：重新运行同一 tag 时先清空旧日志
    logfile.write_text("", encoding="utf-8")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    t_start = time.perf_counter()

    # ---- 语料 ----
    train_text, val_text, corpus_meta = build_corpus(cfg)
    log0("=" * 100, console=False)
    log0(f"# C2G CPU-scale verification  tag={cfg.tag}")
    log0(f"corpus_root:{Path(cfg.corpus_root).resolve()}")
    log0(f"split_seed:{cfg.split_seed if cfg.split_seed is not None else cfg.seed} "
         f"train_seed:{cfg.seed} val_ratio:{cfg.val_ratio}")
    log0(f"corpus_files_total:{corpus_meta['num_files_total']} "
         f"train:{corpus_meta['num_files_train']} val:{corpus_meta['num_files_val']}")
    log0(f"corpus_train_bytes:{corpus_meta['train_bytes_utf8']} "
         f"corpus_val_bytes:{corpus_meta['val_bytes_utf8']}")
    log0(f"val_files:{','.join(corpus_meta['val_file_names'][:8])}")

    # ---- tokenizer ----
    # 只从 train 集取样训练 BPE（防泄漏），两个对照组共用同一样本保证公平
    train_bytes_raw = train_text.encode("utf-8")
    bpe_sample = train_bytes_raw[: cfg.bpe_sample_bytes]
    log0(f"bpe_train_sample_bytes:{len(bpe_sample)} of {len(train_bytes_raw)} "
         f"(sample from train split only, identical across ablation arms)")
    t_bpe = time.perf_counter()
    bpe = train_bpe(bpe_sample, cfg.bpe_merges)
    tok = BPETokenizer(bpe["vocab"], bpe["merges"])
    bpe_seconds = time.perf_counter() - t_bpe
    train_ids = tok.encode(train_text)
    val_ids = tok.encode(val_text)
    train_tokens = torch.tensor(train_ids, dtype=torch.int64)
    val_tokens = torch.tensor(val_ids, dtype=torch.int64)
    byte_lut = tok.byte_lengths()

    # ---- 关键自证：token 字节总数 == 原始 UTF-8 字节数（BPB 口径正确性的前提） ----
    train_bytes_lut = int(byte_lut[np.asarray(train_ids, dtype=np.int64)].sum())
    val_bytes_lut = int(byte_lut[np.asarray(val_ids, dtype=np.int64)].sum())
    assert train_bytes_lut == corpus_meta["train_bytes_utf8"], "token 字节数与原文不符！"
    assert val_bytes_lut == corpus_meta["val_bytes_utf8"], "token 字节数与原文不符！"

    log0(f"bpe_train_seconds:{bpe_seconds:.1f} bpe_merges:{len(bpe['merges'])}")
    log0(f"vocab_size:{tok.vocab_size}")
    log0(f"train_tokens:{train_tokens.numel()} train_ratio:{train_tokens.numel()/corpus_meta['train_bytes_utf8']:.4f} tok/byte")
    log0(f"val_tokens:{val_tokens.numel()} val_ratio:{val_tokens.numel()/corpus_meta['val_bytes_utf8']:.4f} tok/byte")
    log0(f"byte_lut_check:PASS train={train_bytes_lut} val={val_bytes_lut}")

    # ---- 模型 ----
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log0(f"config: layers={cfg.num_layers} dim={cfg.model_dim} heads={cfg.num_heads} "
         f"kv_heads={cfg.num_kv_heads} mlp_mult={cfg.mlp_mult} seq_len={cfg.seq_len} "
         f"batch_tokens={cfg.batch_tokens} steps={cfg.steps} seed={cfg.seed}")
    log0(f"model_params:{n_params}")
    log0(f"tie_embeddings:{cfg.tie_embeddings}")

    # ---- 优化器（矩阵参数走 Muon，其余走 Adam）----
    block_named = list(model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2 and not any(x in n for x in KEEP_FLOAT_PATTERNS)]
    scalar_params = [p for n, p in block_named
                     if p.ndim < 2 or any(x in n for x in KEEP_FLOAT_PATTERNS)]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)
    opt_tok = torch.optim.Adam([{"params": [model.tok_emb.weight], "lr": cfg.embed_lr, "base_lr": cfg.embed_lr}],
                               betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps)
    opt_muon = Muon(matrix_params, lr=cfg.matrix_lr, momentum=cfg.muon_momentum,
                    backend_steps=cfg.muon_backend_steps)
    for g in opt_muon.param_groups:
        g["base_lr"] = cfg.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": cfg.scalar_lr, "base_lr": cfg.scalar_lr}],
                                  betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    stream = TokenStream(train_tokens)
    model.train()

    # ---- 训练循环 ----
    t0 = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % max(cfg.val_every, 1) == 0 or step == cfg.steps:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            vl, vb, tpb, ntok = eval_bpb(model, val_tokens, cfg, byte_lut)
            log0(f"step:{step}/{cfg.steps} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"tokens_per_byte:{tpb:.4f} val_tokens:{ntok} "
                 f"train_time:{elapsed_ms:.0f}ms step_avg:{elapsed_ms/max(step,1):.2f}ms")
        if step == cfg.steps:
            break

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        x, y = make_batch(stream, cfg)
        loss = model(x, y)
        loss.backward()
        scale = lr_scale(step, cfg)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
        for opt in optimizers:
            opt.step()

        if cfg.log_every > 0 and (step + 1) % cfg.log_every == 0:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            log0(f"step:{step+1}/{cfg.steps} train_loss:{loss.item():.4f} lr_scale:{scale:.3f} "
                 f"train_time:{elapsed_ms:.0f}ms step_avg:{elapsed_ms/(step+1):.2f}ms")

    train_seconds = time.perf_counter() - t0
    log0(f"training_seconds:{train_seconds:.1f}")

    # ---- 序列化 + int8+zlib 打包 + 往返验证 ----
    torch.save(model.state_dict(), cfg.artifact_dir / f"{cfg.tag}_model.pt")
    raw_bytes = os.path.getsize(cfg.artifact_dir / f"{cfg.tag}_model.pt")

    quant_obj, qstats = quantize_state_dict_int8(model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    quant_raw = buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    (cfg.artifact_dir / f"{cfg.tag}_model.int8.ptz").write_bytes(quant_blob)

    tok_json = json.dumps(tok.to_jsonable(), separators=(",", ":")).encode("utf-8")
    tok_blob = zlib.compress(tok_json, level=9)
    (cfg.artifact_dir / f"{cfg.tag}_tokenizer.json.zlib").write_bytes(tok_blob)

    code_bytes = len(Path(__file__).read_bytes())
    model_bytes = len(quant_blob)
    tok_bytes = len(tok_blob)
    total_bytes = code_bytes + model_bytes + tok_bytes

    ratio = qstats["baseline_tensor_bytes"] / max(qstats["int8_payload_bytes"], 1)
    log0(f"Serialized model (bf16 raw): {raw_bytes} bytes")
    log0(f"Code size: {code_bytes} bytes")
    log0(f"Tokenizer size (zlib): {tok_bytes} bytes")
    log0(f"Serialized model int8+zlib: {model_bytes} bytes "
         f"(payload:{qstats['int8_payload_bytes']} raw_torch:{len(quant_raw)} payload_ratio:{ratio:.2f}x)")
    log0(f"Total submission size int8+zlib: {total_bytes} bytes")
    log0(f"artifact_cap_bytes:16000000 within_cap:{total_bytes <= 16_000_000}")

    # 往返：把量化后的权重装回模型，重测 BPB（真实测出量化损失）
    model.load_state_dict(dequantize_state_dict_int8(quant_obj), strict=True)
    t_q = time.perf_counter()
    q_vl, q_vb, q_tpb, _ = eval_bpb(model, val_tokens, cfg, byte_lut)
    log0(f"final_int8_zlib_roundtrip val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_q):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")

    # ---- 机器可读结果 ----
    result = {
        "tag": cfg.tag, "seed": cfg.seed, "vocab_size": tok.vocab_size,
        "num_layers": cfg.num_layers, "model_dim": cfg.model_dim, "mlp_mult": cfg.mlp_mult,
        "seq_len": cfg.seq_len, "steps": cfg.steps, "batch_tokens": cfg.batch_tokens,
        "model_params": n_params,
        "train_tokens": int(train_tokens.numel()), "val_tokens": int(val_tokens.numel()),
        "train_bytes": corpus_meta["train_bytes_utf8"], "val_bytes": corpus_meta["val_bytes_utf8"],
        "train_tokens_per_byte": train_tokens.numel() / corpus_meta["train_bytes_utf8"],
        "val_tokens_per_byte": val_tokens.numel() / corpus_meta["val_bytes_utf8"],
        "val_bpb": q_vb, "val_loss": q_vl,
        "val_bpb_pre_quant": float(vb), "val_loss_pre_quant": float(vl),
        "tokens_per_byte_eval": q_tpb,
        "bytes_code": code_bytes, "bytes_model_int8_zlib": model_bytes,
        "bytes_tokenizer_zlib": tok_bytes, "bytes_total": total_bytes,
        "within_16mb_cap": total_bytes <= 16_000_000,
        "training_seconds": round(train_seconds, 1),
        "bpe_train_seconds": round(bpe_seconds, 1),
        "hardware": "CPU (no GPU available in this environment)",
        "note": ("CPU-scale verification run on local corpus; NOT an 8xH100/FineWeb "
                 "leaderboard number. Same metric definition and packing pipeline as "
                 "official train_gpt.py."),
    }
    (cfg.artifact_dir / f"{cfg.tag}_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log0(f"result_json:{cfg.artifact_dir / f'{cfg.tag}_result.json'}")
    log0(f"total_seconds:{time.perf_counter() - t_start:.1f}")


if __name__ == "__main__":
    main()
