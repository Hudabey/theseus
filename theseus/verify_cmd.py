"""`theseus verify <repo>` — checkpoint integrity oracle.

Verifies, without downloading weights: upload completeness (index vs hub),
header/offset consistency per shard, MXFP4 blocks/scales pairing, expert-set
completeness per layer, and parameter count. ✓/✗ per check; exit 0 only if all
hard checks pass."""
from __future__ import annotations

import collections
import re
import sys

from . import census, hub


def run(repo: str) -> int:
    print(f"theseus verify — {repo}\n" + "─" * 56)
    fails = 0

    def check(ok: bool, msg: str, hard: bool = True):
        nonlocal fails
        print(f"  {'✓' if ok else ('✗' if hard else '⚠')} {msg}")
        if not ok and hard:
            fails += 1

    files = hub.list_files(repo)
    idx = hub.get_json(repo, "model.safetensors.index.json", optional=True)
    if idx is None:
        check("model.safetensors" in files, "single-file checkpoint present (no index)")
        named = {"model.safetensors"} if "model.safetensors" in files else set()
    else:
        named = set(idx["weight_map"].values())
        on_hub = {f for f in files if f.endswith(".safetensors")}
        missing = sorted(named - on_hub)
        check(not missing, f"upload complete: index names {len(named)} shards, "
                           f"{len(named) - len(missing)} on hub"
                           + (f" — MISSING {missing[:3]}…" if missing else ""))
        extra = sorted(on_hub - named)
        check(not extra, f"no shard files outside the index"
                         + (f" ({len(extra)} extra: {extra[:3]}…)" if extra else ""),
              hard=False)

    rows = hub.fetch_all_headers(
        repo, progress=lambda s: print(s, end="\r", file=sys.stderr))
    print(file=sys.stderr)
    check(len(rows) > 0, f"shard headers readable: {len(rows)} tensors")

    # offsets per shard: contiguous coverage, no overlap, end == file size
    bad_shards = []
    by_shard = collections.defaultdict(list)
    for r in rows:
        by_shard[r["shard"]].append(r)
    for shard, rs in by_shard.items():
        spans = sorted(r["offsets"] for r in rs)
        end = 0
        ok = True
        for a, b in spans:
            if a != end or b < a:
                ok = False
                break
            end = b
        size = files.get(shard)
        if ok and size and end + 8 + rs[0]["header_len"] != size:
            ok = False
        if not ok:
            bad_shards.append(shard)
    check(not bad_shards, f"tensor offsets contiguous and match shard sizes "
                          f"({len(by_shard)} shards)"
                          + (f" — BAD: {bad_shards[:3]}" if bad_shards else ""))

    # MXFP4-style pairing: GPT-OSS naming (*_blocks/*_scales) and
    # compressed-tensors naming (*.weight_packed/*.weight_scale)
    PAIRS = (("_blocks", "_scales"), ("weight_packed", "weight_scale"))
    blocks, scales = {}, {}
    for r in rows:
        for bsuf, ssuf in PAIRS:
            if r["name"].endswith(bsuf):
                blocks[r["name"]] = (r, bsuf, ssuf)
            elif r["name"].endswith(ssuf):
                scales[r["name"]] = r
    orphans, mismatched = [], []
    for name, (b, bsuf, ssuf) in blocks.items():
        s = scales.get(name[: -len(bsuf)] + ssuf)
        if s is None:
            orphans.append(name)
        elif b["shape"][:-1] != s["shape"][:-1] or b["shape"][-1] != s["shape"][-1] * 16:
            mismatched.append(name)
    if blocks:
        check(not orphans and not mismatched,
              f"quantized pairs consistent: {len(blocks)} blocks/scales pairs, "
              f"group size 32 (blocks ne0 = 16 bytes × scales ne0)"
              + (f" — orphans {orphans[:2]} mismatched {mismatched[:2]}" if orphans or mismatched else ""))
    else:
        print(f"  · no *_blocks/*_scales quantized tensors (float checkpoint)")

    # expert completeness per layer
    exp = collections.defaultdict(set)
    for r in rows:
        m = re.search(r"layers\.(\d+)\..*experts\.(\d+)\.", r["name"])
        if m:
            exp[int(m.group(1))].add(int(m.group(2)))
    if exp:
        counts = {n for s in exp.values() for n in (len(s),)}
        ragged = [il for il, s in exp.items() if len(s) != max(counts)]
        check(len(counts) == 1,
              f"expert sets complete: {max(counts)} experts on each of {len(exp)} MoE layers"
              + (f" — ragged layers {ragged[:5]}" if ragged else ""))

    buckets, unmatched = census.classify(rows)
    check(not unmatched, f"all {len(rows)} tensor names classified into known families",
          hard=False)
    if unmatched:
        for r in unmatched[:10]:
            print(f"      unclassified: {r['name']}  {r['dtype']}  {r['shape']}")

    params = census.param_count(rows)
    print(f"  · parameter count: {params:,} ({params / 1e12:.2f}T) [CALCULATED]")

    print("\n" + ("VERIFIED" if fails == 0 else f"FAILED ({fails} hard check(s))"))
    return 0 if fails == 0 else 1


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: theseus verify <org/repo>")
        return 2
    return run(args[0])
