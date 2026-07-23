#!/usr/bin/env python
"""Dump every tensor's name/dtype/shape/byte-offsets WITHOUT downloading weights.

The safetensors index JSON maps names to shard files only; shapes/dtypes/offsets live
in each shard's header (8-byte little-endian length + JSON). This fetches just those
headers via HTTP Range requests and writes tensors.json.

Usage: fetch_headers.py <repo_id> [-o tensors.json]
"""
import argparse
import json
import os
import struct

import requests


def hf_headers() -> dict:
    h = {}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def resolve(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def fetch_range(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={**hf_headers(), "Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    if r.status_code != 206 and len(r.content) != end - start + 1:
        raise RuntimeError(f"range request not honored for {url} (status {r.status_code})")
    return r.content


def shard_header(repo: str, shard: str) -> dict:
    url = resolve(repo, shard)
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    return json.loads(fetch_range(url, 8, 7 + n)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("-o", "--out", default="tensors.json")
    args = ap.parse_args()

    r = requests.get(resolve(args.repo, "model.safetensors.index.json"),
                     headers=hf_headers(), timeout=60)
    if r.status_code == 404:
        shards = ["model.safetensors"]
    else:
        r.raise_for_status()
        shards = sorted(set(json.loads(r.content)["weight_map"].values()))

    rows = []
    for i, shard in enumerate(shards):
        hdr, hlen = shard_header(args.repo, shard)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            rows.append({
                "name": name,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "shard": shard,
                "header_len": hlen,
                "offsets": meta["data_offsets"],
            })
        print(f"[headers] {shard} ({i + 1}/{len(shards)}): {len(hdr) - ('__metadata__' in hdr)} tensors")

    rows.sort(key=lambda x: x["name"])
    with open(args.out, "w") as f:
        json.dump(rows, f)
    from collections import Counter
    dt = Counter(x["dtype"] for x in rows)
    print(f"[headers] wrote {len(rows)} tensors from {len(shards)} shards -> {args.out}")
    print(f"[headers] dtypes: {dict(dt)}")


if __name__ == "__main__":
    main()
