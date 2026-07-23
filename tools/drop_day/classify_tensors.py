#!/usr/bin/env python
"""Bucket every tensor name from tensors.json; the UNMATCHED bucket is the discovery
list (SiTU / LatentMoE structure / MLA gate / anything nobody predicted).

Usage: classify_tensors.py [tensors.json] [--show N]
Writes unmatched.json next to the input.
"""
import argparse
import collections
import json
import re

# first match wins; patterns = expected mapping from recon 01 §5 (kimi_linear.py) +
# recon 04 §4 (attnres) + standard HF blocks
BUCKETS = [
    ("attnres", r"res_proj|res_norm|attn_res|mlp_res"),
    ("kda",     r"A_log|dt_bias|dt_proj|f_[ab]_proj|g_[ab]_proj|\bb_proj|conv1d|beta"),
    ("mla",     r"q_a_|q_b_|kv_a_|kv_b_|q_proj|k_proj|v_proj|o_proj|attn.*gate|kv_norm|q_norm"),
    ("moe",     r"experts|shared_expert|router|\bgate\.weight|e_score"),
    ("mlp",     r"gate_proj|up_proj|down_proj"),
    ("norms",   r"norm"),
    ("embed",   r"embed|lm_head"),
    ("vision",  r"vision|visual|image|patch|merger"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tensors", nargs="?", default="tensors.json")
    ap.add_argument("--show", type=int, default=80)
    args = ap.parse_args()

    rows = json.load(open(args.tensors))
    buckets = collections.defaultdict(list)
    unmatched = []
    for row in rows:
        for bucket, pat in BUCKETS:
            if re.search(pat, row["name"]):
                buckets[bucket].append(row)
                break
        else:
            unmatched.append(row)

    for bucket, _ in BUCKETS:
        n = len(buckets[bucket])
        ex = buckets[bucket][0]["name"] if n else ""
        print(f"{bucket:8s} {n:6d}  {ex}")
    print(f"{'UNMATCH':8s} {len(unmatched):6d}")
    for row in unmatched[: args.show]:
        print(f"   {row['name']}  {row['dtype']}  {row['shape']}")

    out = args.tensors.replace("tensors", "unmatched")
    with open(out, "w") as f:
        json.dump(unmatched, f, indent=1)
    print(f"[classify] wrote {len(unmatched)} unmatched -> {out}")


if __name__ == "__main__":
    main()
