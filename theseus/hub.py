"""Hugging Face hub access: file listings, safetensors index, and shard headers
via HTTP Range requests — tensor names/shapes/dtypes/offsets with zero weight
bytes downloaded. Same mechanics as tools/drop_day/{fetch_headers,check_complete}.py,
rehearsed 2026-07-22 against moonshotai/Kimi-Linear-48B-A3B-Instruct."""
from __future__ import annotations

import json
import os
import struct

import requests

HUB = "https://huggingface.co"


def hf_headers() -> dict:
    h = {}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def resolve(repo: str, path: str) -> str:
    return f"{HUB}/{repo}/resolve/main/{path}"


def get_json(repo: str, path: str, *, optional: bool = False):
    r = requests.get(resolve(repo, path), headers=hf_headers(), timeout=60)
    if r.status_code == 404 and optional:
        return None
    r.raise_for_status()
    return json.loads(r.content)


def list_files(repo: str) -> dict[str, int]:
    """path -> size in bytes for every file in the repo."""
    files: dict[str, int] = {}
    url = f"{HUB}/api/models/{repo}/tree/main"
    cursor = None
    while True:
        params = {"limit": 1000, **({"cursor": cursor} if cursor else {})}
        r = requests.get(url, params=params, headers=hf_headers(), timeout=30)
        r.raise_for_status()
        batch = r.json()
        for f in batch:
            files[f["path"]] = f.get("size", 0)
        cursor = r.headers.get("x-next-cursor") or (r.links.get("next", {}).get("url"))
        if not cursor or not batch:
            break
        if cursor.startswith("http"):
            url, cursor = cursor, None
    return files


def shard_names(repo: str) -> list[str]:
    idx = get_json(repo, "model.safetensors.index.json", optional=True)
    if idx is None:
        return ["model.safetensors"]
    return sorted(set(idx["weight_map"].values()))


def fetch_range(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={**hf_headers(), "Range": f"bytes={start}-{end}"},
                     timeout=60)
    r.raise_for_status()
    if r.status_code != 206 and len(r.content) != end - start + 1:
        raise RuntimeError(f"range request not honored for {url} (status {r.status_code})")
    return r.content


def shard_header(repo: str, shard: str) -> tuple[dict, int]:
    url = resolve(repo, shard)
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    return json.loads(fetch_range(url, 8, 7 + n)), n


def fetch_all_headers(repo: str, progress=None) -> list[dict]:
    """One row per tensor: name/dtype/shape/shard/offsets. No weight bytes."""
    shards = shard_names(repo)
    rows = []
    for i, shard in enumerate(shards):
        hdr, hlen = shard_header(repo, shard)
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
        if progress:
            progress(f"  headers {i + 1}/{len(shards)}: {shard}")
    rows.sort(key=lambda x: x["name"])
    return rows
