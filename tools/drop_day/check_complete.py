#!/usr/bin/env python
"""Completeness gate: refuse to start recon against a partially-uploaded repo.

Checks (1) model.safetensors.index.json exists, (2) every shard the index names is
present on the hub, (3) no shard-like files on the hub are missing from the index.
Exit 0 = safe to proceed; exit 1 = wait.

Usage: check_complete.py <repo_id>
"""
import json
import os
import sys

import requests


def main() -> None:
    repo = sys.argv[1]
    hdrs = {}
    if os.environ.get("HF_TOKEN"):
        hdrs["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

    files = set()
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    cursor = None
    while True:
        params = {"limit": 1000, **({"cursor": cursor} if cursor else {})}
        r = requests.get(url, params=params, headers=hdrs, timeout=30)
        r.raise_for_status()
        batch = r.json()
        files |= {f["path"] for f in batch}
        cursor = r.headers.get("x-next-cursor") or (r.links.get("next", {}).get("url"))
        if not cursor or not batch:
            break
        if cursor.startswith("http"):  # link-style cursor: query param embedded
            url, cursor = cursor, None

    if "model.safetensors.index.json" not in files:
        if "model.safetensors" in files:
            print("[complete] single-file checkpoint (no index); proceeding")
            return
        print("[complete] FAIL: model.safetensors.index.json not on hub yet")
        sys.exit(1)

    r = requests.get(
        f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json",
        headers=hdrs, timeout=60,
    )
    r.raise_for_status()
    idx = json.loads(r.content)
    named = set(idx["weight_map"].values())
    on_hub = {f for f in files if f.endswith(".safetensors")}

    missing = sorted(named - on_hub)
    extra = sorted(on_hub - named)
    print(f"[complete] index names {len(named)} shards; hub has {len(on_hub)} .safetensors files")
    if missing:
        print(f"[complete] FAIL: {len(missing)} shards named in index but absent: {missing[:5]} …")
        sys.exit(1)
    if extra:
        print(f"[complete] note: files on hub not in index (vision/other?): {extra[:10]}")
    print("[complete] OK — safe to start Step 1")


if __name__ == "__main__":
    main()
