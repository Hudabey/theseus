#!/usr/bin/env python
"""Watch the Moonshot HF org for a K3 release — by STATE, not just by repo creation.

A repo can be created empty (or flipped private→public) and populated later, so
watching only for new repo ids is not enough. Per matching repo this tracks:
  - existence (new id appearing at all),
  - revision SHA changes (any commit),
  - appearance of config.json,
  - appearance of model.safetensors.index.json  <- the actionable moment: exits here.

Honors HF_TOKEN (gated/private repos the token can see).

Usage: watch_release.py [--interval 180] [--author moonshotai] [--pattern k3]
"""
import argparse
import datetime
import os
import re
import time

import requests

API = "https://huggingface.co/api"


def hf_headers() -> dict:
    h = {}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def list_repo_ids(author: str) -> set[str]:
    r = requests.get(f"{API}/models", params={"author": author, "limit": 200},
                     headers=hf_headers(), timeout=30)
    r.raise_for_status()
    return {m["id"] for m in r.json()}


def repo_state(repo_id: str) -> dict | None:
    """sha + tracked-file presence for one repo; None if unreadable (404/private)."""
    r = requests.get(f"{API}/models/{repo_id}", headers=hf_headers(), timeout=30)
    if r.status_code != 200:
        return None
    j = r.json()
    files = {s["rfilename"] for s in j.get("siblings", [])}
    return {
        "sha": j.get("sha"),
        "config": "config.json" in files,
        "index": "model.safetensors.index.json" in files,
    }


def loud(msg: str, times: int = 5) -> None:
    for _ in range(times):
        print(f"\a{'=' * 60}\n{msg}\n{'=' * 60}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--author", default="moonshotai")
    ap.add_argument("--pattern", default=r"k3", help="case-insensitive regex on repo id")
    args = ap.parse_args()
    pat = re.compile(args.pattern, re.IGNORECASE)

    known: dict[str, dict] = {}
    ids = list_repo_ids(args.author)
    for rid in sorted(rid for rid in ids if pat.search(rid)):
        known[rid] = repo_state(rid) or {}
    print(f"[watch] baseline: {len(ids)} repos under {args.author}; "
          f"matching tracked: {list(known) or 'none'} "
          f"(token: {'yes' if os.environ.get('HF_TOKEN') else 'no'})")

    while True:
        time.sleep(args.interval)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            ids = list_repo_ids(args.author)
        except Exception as e:  # noqa: BLE001 — keep polling through transient errors
            print(f"[{ts}] org poll failed ({e}); retrying")
            continue

        events = []
        for rid in sorted(rid for rid in ids if pat.search(rid)):
            try:
                st = repo_state(rid)
            except Exception as e:  # noqa: BLE001
                print(f"[{ts}] state poll failed for {rid} ({e})")
                continue
            if st is None:
                continue
            old = known.get(rid)
            if old is None:
                events.append(f"NEW MATCHING REPO {rid} (state: {st})")
            else:
                if st.get("sha") != old.get("sha"):
                    events.append(f"{rid}: revision changed {old.get('sha')} -> {st.get('sha')}")
                if st.get("config") and not old.get("config"):
                    events.append(f"{rid}: config.json APPEARED")
                if st.get("index") and not old.get("index"):
                    events.append(f"{rid}: model.safetensors.index.json APPEARED")
            known[rid] = st

        for msg in events:
            loud(f"[{ts}] {msg}")

        ready = [rid for rid, st in known.items() if st.get("index")]
        if ready:
            loud(f"[{ts}] INDEX PRESENT — start the runbook (Step 0c first):")
            print(f"export REPO={ready[0]}")
            return
        if not events:
            print(f"[{ts}] no change ({len(ids)} repos, {len(known)} matching tracked)")


if __name__ == "__main__":
    main()
