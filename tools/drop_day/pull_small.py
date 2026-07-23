#!/usr/bin/env python
"""Pull individual small tensors by name pattern via HTTP Range requests (no shard
download) and print dtype-correct norms. BF16 is decoded properly:
uint16 -> uint32 << 16 -> view float32 (bfloat16 is the top half of an IEEE float32).

Usage:
  pull_small.py <repo_id> --match res_proj              [--tensors tensors.json] [--save]
  pull_small.py --self-test                             # offline BF16 unit test
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import requests


def hf_headers() -> dict:
    h = {}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def fetch_range(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={**hf_headers(), "Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    return r.content


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


def decode(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    """Return a float32 (or uint8 for packed quants) array of `shape`."""
    if dtype == "BF16":
        arr = bf16_to_f32(raw)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F64":
        arr = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    elif dtype in ("U8", "I8", "F8_E4M3", "F8_E5M2", "F8_E8M0"):
        arr = np.frombuffer(raw, dtype=np.uint8)  # packed/quant bytes: no float view
    else:
        raise ValueError(f"unhandled safetensors dtype {dtype}")
    return arr.reshape(shape)


DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # this tool is for SMALL tensors only


def pull(repo: str, row: dict, save: bool, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    a0, b0 = row["offsets"]
    size = b0 - a0
    if max_bytes > 0 and size > max_bytes:
        print(f"REFUSED {row['name']}: {size / 1e6:.1f} MB > --max-bytes "
              f"{max_bytes / 1e6:.1f} MB (a broad pattern like 'experts.0' can match "
              f"huge matrices; raise --max-bytes explicitly if intended)")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{row['shard']}"
    hlen = row.get("header_len")
    if hlen is None:
        hlen = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    a, b = row["offsets"]
    raw = fetch_range(url, 8 + hlen + a, 8 + hlen + b - 1)
    arr = decode(raw, row["dtype"], row["shape"])
    if arr.dtype == np.uint8:
        print(f"{row['name']}  {row['dtype']} {row['shape']}  "
              f"bytes={arr.size} min={arr.min()} max={arr.max()} (packed — no norm)")
    else:
        arr64 = arr.astype(np.float64)
        print(f"{row['name']}  {row['dtype']} {row['shape']}  "
              f"norm={float(np.linalg.norm(arr64)):.6g} "
              f"mean={float(arr64.mean()):.4g} absmax={float(np.abs(arr64).max()):.4g}")
    if save:
        fn = row["name"].replace("/", "_").replace(".", "_") + ".npy"
        np.save(fn, arr)
        print(f"  saved -> {fn}")


def self_test() -> None:
    # values exactly representable in bfloat16, plus signs and a subnormal-free zero
    vals = np.array([1.5, -2.0, 0.25, 0.0, -96.0, 2.0 ** 100], dtype=np.float32)
    u16 = (vals.view(np.uint32) >> 16).astype(np.uint16)  # truncate to bf16 (exact here)
    raw = u16.astype("<u2").tobytes()
    out = decode(raw, "BF16", [2, 3])
    assert out.dtype == np.float32, out.dtype
    assert np.array_equal(out.reshape(-1), vals), (out.reshape(-1), vals)
    n = float(np.linalg.norm(out.astype(np.float64)))
    assert n > 0 and np.isfinite(n), n
    # regression guard for the original bug: BF16 must NOT decode to zeros
    assert not np.allclose(out, 0.0)
    # F16 path sanity
    h = np.array([1.0, -0.5], dtype=np.float16)
    out16 = decode(h.tobytes(), "F16", [2])
    assert np.array_equal(out16, np.array([1.0, -0.5], dtype=np.float32))
    print("[self-test] OK: BF16 round-trip exact, norm =", n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?")
    ap.add_argument("--tensors", default="tensors.json")
    ap.add_argument("--match", help="regex on tensor name")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                    help=f"refuse tensors larger than this (default {DEFAULT_MAX_BYTES}; 0 = no limit)")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not (args.repo and args.match):
        ap.error("need <repo> and --match (or --self-test)")

    rows = [r for r in json.load(open(args.tensors)) if re.search(args.match, r["name"])]
    if not rows:
        print(f"[pull] no tensor matches {args.match!r}", file=sys.stderr)
        sys.exit(1)
    for row in rows[: args.limit]:
        pull(args.repo, row, args.save, args.max_bytes)
    if len(rows) > args.limit:
        print(f"[pull] … {len(rows) - args.limit} more matches not pulled (raise --limit)")


if __name__ == "__main__":
    main()
