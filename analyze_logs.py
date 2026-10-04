#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_logs.py —— C2G 训练日志 / 实测结果分析工具

两种用法：
  1) 解析训练日志（官方 train_gpt.py 或本包 verify_cpu.py 产出，格式一致）
       python analyze_logs.py logs/*.log
  2) 汇总真实实测结果 JSON，直接生成消融对比表
       python analyze_logs.py --results artifacts
       python analyze_logs.py --results artifacts --markdown

指标口径与官方 train_gpt.py 完全一致：
    val_bpb = (val_loss / ln2) * (token_count / byte_count)
    即 final_int8_zlib_roundtrip_exact 行报告的值（量化往返后的权威值）。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path


# ----------------------------------------------------------------------------- 日志解析
def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "file": path.name,
        "run_id": _grab(r"logs/(\S+)\.txt", text) or _grab(r"tag=(\S+)", text),
        "val_bpb": _float(_grab(r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)", text)),
        "val_loss": _float(_grab(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+)", text)),
        "pre_quant_bpb": _float(_last(r"val_bpb:([\d.]+)", text)),
        "train_time_ms": _int(_last(r"train_time:(\d+)ms", text)),
        "steps": _int(_last(r"step:(\d+)/\d+", text)),
        "submission_size": _int(_grab(r"Total submission size int8\+zlib: (\d+) bytes", text)),
        "model_params": _int(_grab(r"model_params:(\d+)", text)),
        "peak_mem_mib": _int(_grab(r"peak memory allocated: (\d+) MiB", text)),
        "vocab_size": _int(_grab(r"vocab_size:(\d+)", text)),
        "tokens_per_byte": _float(_last(r"tokens_per_byte:([\d.]+)", text)),
    }


def _grab(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def _last(pattern: str, text: str) -> str:
    ms = re.findall(pattern, text)
    return ms[-1] if ms else ""


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


LOG_COLS = [
    ("file", "日志文件"), ("run_id", "run_id"), ("vocab_size", "vocab"), ("steps", "steps"),
    ("train_time_ms", "train_time(ms)"), ("val_loss", "val_loss"), ("val_bpb", "val_bpb"),
    ("tokens_per_byte", "tok/byte"), ("submission_size", "submission(bytes)"),
    ("model_params", "参数量"), ("peak_mem_mib", "峰值显存(MiB)"),
]


def mode_logs(patterns: list[str]) -> int:
    files: list[Path] = []
    for p in patterns:
        files.extend(Path(x) for x in glob.glob(p))
    files = sorted(set(files))
    if not files:
        print("未找到日志文件，用法：python analyze_logs.py logs/*.log", file=sys.stderr)
        return 1
    rows = [parse_log(f) for f in files]
    print("\t".join(label for _, label in LOG_COLS))
    for r in rows:
        print("\t".join(_fmt(r[k]) for k, _ in LOG_COLS))
    scored = [r for r in rows if r["val_bpb"] is not None]
    if scored:
        print("\n# BPB 汇总（越低越好）")
        for r in sorted(scored, key=lambda x: x["val_bpb"]):
            print(f"  {r['val_bpb']:.4f}  {r['file']}  (loss={_fmt(r['val_loss'])}, "
                  f"{r['submission_size']} bytes)")
    return 0


# ----------------------------------------------------------------------------- 结果汇总
RESULT_COLS = [
    ("tag", "run"), ("vocab_size", "vocab"), ("val_tokens_per_byte", "val tok/byte"),
    ("model_params", "params"), ("val_loss", "val_loss"), ("val_bpb", "val_bpb"),
    ("train_tokens", "train_tokens"), ("train_bytes", "train_bytes"),
    ("bytes_total", "artifact(bytes)"), ("training_seconds", "train(s)"),
]


def mode_results(results_dir: Path, as_markdown: bool) -> int:
    files = sorted(results_dir.glob("*_result.json"))
    if not files:
        print(f"未找到 {results_dir}/*_result.json", file=sys.stderr)
        return 1
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows.sort(key=lambda r: r.get("val_bpb", 9e9))

    # 基准 = 词表最小的那一臂（消融的对照组）
    base = min(rows, key=lambda r: r.get("vocab_size", 9e9))
    base_bpb = base.get("val_bpb")

    if as_markdown:
        print("| run | vocab | val tok/byte | val_loss | val_bpb | Δ vs 小词表 | artifact(bytes) | train(s) |")
        print("|-----|------:|-------------:|---------:|--------:|-----------:|----------------:|---------:|")
        for r in rows:
            d = (r.get("val_bpb", 0) - base_bpb) if base_bpb else 0
            print(f"| `{r['tag']}` | {r['vocab_size']} | {r.get('val_tokens_per_byte', 0):.4f} | "
                  f"{r.get('val_loss', 0):.4f} | **{r.get('val_bpb', 0):.4f}** | {d:+.4f} | "
                  f"{r.get('bytes_total', 0):,} | {r.get('training_seconds', 0)} |")
        print()
        print(f"- 基准（最小词表 `{base['tag']}`）val_bpb = {base_bpb:.4f}")
        best = rows[0]
        print(f"- 最优 `{best['tag']}` val_bpb = {best.get('val_bpb'):.4f} "
              f"（相对基准 {best.get('val_bpb') - base_bpb:+.4f}）")
        print(f"- 硬件/口径：{best.get('hardware')}")
        print(f"- 说明：{best.get('note')}")
    else:
        print("\t".join(label for _, label in RESULT_COLS))
        for r in rows:
            print("\t".join(_fmt(r.get(k)) for k, _ in RESULT_COLS))

    print("\n# 完整结果")
    for r in rows:
        print(f"  {r['tag']}: bpb={r.get('val_bpb')} loss={r.get('val_loss')} "
              f"vocab={r.get('vocab_size')} params={r.get('model_params')} "
              f"bytes={r.get('bytes_total')} within_cap={r.get('within_16mb_cap')}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="C2G 日志/结果分析")
    ap.add_argument("logs", nargs="*", help="日志文件 glob（默认 logs/*.log）")
    ap.add_argument("--results", default=None, help="汇总 *_result.json 的目录（如 artifacts）")
    ap.add_argument("--markdown", action="store_true", help="以 Markdown 表格输出结果汇总")
    args = ap.parse_args(argv[1:])

    if args.results:
        return mode_results(Path(args.results), args.markdown)
    return mode_logs(args.logs or ["logs/*.log"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
