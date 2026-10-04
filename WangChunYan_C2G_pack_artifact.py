#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WangChunYan_C2G_pack_artifact.py
================================================================================
把真实训练产物打包成官方要求的 <16MB 提交 artifact（.tar.gz）。

官方口径（README FAQ）：
    submission artifact = code bytes + compressed model bytes
    cap = 16,000,000 字节（十进制 16MB，不是 16MiB）
    且 artifact 必须自包含、可复现、评估时不得联网。

本脚本做的事：
    1. 扫描 artifacts/*_result.json，挑出 val_bpb 最低的那次真实运行
    2. 把「代码 + int8+zlib 模型 + zlib tokenizer + 结果元数据」打进 tar.gz
    3. 校验总字节数是否在 16,000,000 之内，并写出 MANIFEST

用法：
    python WangChunYan_C2G_pack_artifact.py
    python WangChunYan_C2G_pack_artifact.py --tag v2048_s1337
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

CAP_BYTES = 16_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_best(artifact_dir: Path, tag: str | None) -> tuple[dict, Path]:
    results = sorted(artifact_dir.glob("*_result.json"))
    if not results:
        raise SystemExit(f"未找到结果文件：{artifact_dir}/*_result.json（请先跑 verify_cpu.py）")
    if tag:
        for r in results:
            if r.stem == f"{tag}_result":
                return json.loads(r.read_text(encoding="utf-8")), r
        raise SystemExit(f"未找到 tag={tag} 的结果文件")
    best, best_path = None, None
    for r in results:
        d = json.loads(r.read_text(encoding="utf-8"))
        if best is None or d.get("val_bpb", 9e9) < best.get("val_bpb", 9e9):
            best, best_path = d, r
    return best, best_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="指定打包哪一次运行（默认取 val_bpb 最低）")
    ap.add_argument("--root", default=".", help="交付包根目录")
    ap.add_argument("--out", default=None, help="输出 tar.gz 路径")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    artifact_dir = root / "artifacts"
    result, result_path = pick_best(artifact_dir, args.tag)
    tag = result["tag"]

    out_path = Path(args.out) if args.out else root / "WangChunYan_C2G_submission.tar.gz"

    # ---- 待打包成员：(归档内路径, 磁盘路径) ----
    members: list[tuple[str, Path]] = [
        ("code/WangChunYan_C2G_verify_cpu.py", root / "WangChunYan_C2G_verify_cpu.py"),
        ("code/WangChunYan_C2G_train_gpt.py", root / "WangChunYan_C2G_train_gpt.py"),
        ("model/model.int8.ptz", artifact_dir / f"{tag}_model.int8.ptz"),
        ("model/tokenizer.json.zlib", artifact_dir / f"{tag}_tokenizer.json.zlib"),
        ("result.json", result_path),
    ]
    missing = [str(p) for _, p in members if not p.is_file()]
    if missing:
        raise SystemExit("缺少待打包文件：\n  " + "\n  ".join(missing))

    # ---- 写 MANIFEST ----
    manifest = {
        "challenge": "C2G Parameter Golf",
        "author": "WangChunYan",
        "student_id": "2023105110151",
        "packed_tag": tag,
        "val_bpb": result.get("val_bpb"),
        "val_loss": result.get("val_loss"),
        "vocab_size": result.get("vocab_size"),
        "model_params": result.get("model_params"),
        "hardware": result.get("hardware"),
        "note": result.get("note"),
        "artifact_cap_bytes": CAP_BYTES,
        "member_sha256": {arc: sha256(p) for arc, p in members},
        "member_bytes": {arc: p.stat().st_size for arc, p in members},
    }
    manifest_path = artifact_dir / f"{tag}_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    members.append(("MANIFEST.json", manifest_path))

    # ---- 写 tar.gz（固定 mtime，保证可复现的字节数）----
    with tarfile.open(out_path, "w:gz") as tar:
        for arc, p in members:
            ti = tar.gettarinfo(str(p), arcname=arc)
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            with open(p, "rb") as f:
                tar.addfile(ti, f)

    total = out_path.stat().st_size
    code_bytes = sum(p.stat().st_size for a, p in members if a.startswith("code/"))
    model_bytes = sum(p.stat().st_size for a, p in members if a.startswith("model/"))

    print(f"artifact           : {out_path.name}")
    print(f"packed tag         : {tag}")
    print(f"val_bpb (measured) : {result.get('val_bpb')}")
    print(f"code bytes         : {code_bytes}")
    print(f"model bytes        : {model_bytes}")
    print(f"tar.gz total bytes : {total}")
    print(f"cap                : {CAP_BYTES}")
    print(f"within cap         : {total <= CAP_BYTES}  (headroom {CAP_BYTES - total} bytes)")
    print(f"MANIFEST           : {manifest_path.name}")


if __name__ == "__main__":
    main()
