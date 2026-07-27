"""Tensor census: classify names into architectural families, count parameters,
sum bytes. Buckets identical to tools/drop_day/classify_tensors.py (validated:
0 unmatched on Kimi-Linear's 20,493 real tensors)."""
from __future__ import annotations

import collections
import math
import re

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

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def classify(rows: list[dict]):
    buckets = collections.defaultdict(list)
    unmatched = []
    for row in rows:
        for bucket, pat in BUCKETS:
            if re.search(pat, row["name"]):
                buckets[bucket].append(row)
                break
        else:
            unmatched.append(row)
    return buckets, unmatched


def numel(row: dict) -> int:
    return math.prod(row["shape"]) if row["shape"] else 1


def nbytes(row: dict) -> int:
    return row["offsets"][1] - row["offsets"][0]


def param_count(rows: list[dict]) -> int:
    """Logical parameter count. MXFP4-style packed U8 `*_blocks` tensors hold two
    4-bit values per byte; their `*_scales` companions are metadata, not params."""
    total = 0
    for row in rows:
        if row["name"].endswith(("_scales", ".scales")):
            continue
        n = numel(row)
        if row["dtype"] == "U8" and row["name"].endswith(("_blocks", ".blocks")):
            n *= 2
        total += n
    return total


def layer_census(rows: list[dict]) -> dict[int, set[str]]:
    """layer index -> set of bucket names present on that layer."""
    per_layer: dict[int, set[str]] = collections.defaultdict(set)
    for row in rows:
        m = re.search(r"layers\.(\d+)\.", row["name"])
        if not m:
            continue
        il = int(m.group(1))
        for bucket, pat in BUCKETS:
            if re.search(pat, row["name"]):
                per_layer[il].add(bucket)
                break
    return per_layer
