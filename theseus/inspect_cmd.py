"""`theseus inspect <repo>` — can my hardware run this model?

Reads config.json + shard headers (HTTP Range requests, zero weight bytes) and
reports architecture, true download size, and feasibility against the local
machine. Every number is labeled MEASURED (read from the checkpoint/machine) or
CALCULATED (derived); nothing is estimated from thin air."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys

from . import census, hub

GB = 1024 ** 3


def local_ram_bytes() -> int | None:
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def local_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10).decode()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 4:
        return f"{n / 1024 ** 4:.2f} TB"
    return f"{n / GB:.1f} GB"


def run(repo: str) -> int:
    print(f"theseus inspect — {repo}\n" + "─" * 56)

    cfg = hub.get_json(repo, "config.json", optional=True)
    if cfg is None:
        print("✗ config.json not found — repo empty, gated without token, or mid-upload")
        return 1

    # vision-language wrappers nest the LM under text_config (e.g. Kimi-K3)
    tcfg = cfg.get("text_config") or {}

    def hp(*keys, default=None):
        for k in keys:
            for c in (cfg, tcfg):
                if c.get(k) is not None:
                    return c[k]
        return default

    arch = ", ".join(cfg.get("architectures", ["?"]))
    n_layer = hp("num_hidden_layers", "n_layer")
    ctx = hp("max_position_embeddings", "model_max_length", default="?")
    print(f"architecture:      {arch}")
    print(f"layers:            {n_layer}   vocab: {hp('vocab_size')}   context: {ctx}")
    n_experts = hp("n_routed_experts", "num_local_experts", "num_experts")
    topk = hp("num_experts_per_tok", "num_experts_per_token", "moe_topk", "moe_top_k", "top_k")
    if n_experts:
        print(f"MoE:               {n_experts} routed experts, {topk or '?'} active/token, "
              f"{hp('num_shared_experts', 'n_shared_experts', default=0)} shared")
    quant = (cfg.get("quantization_config") or {}).get("quant_method")
    if quant:
        print(f"quantization:      {quant} (native release format)")

    print("\nfetching tensor inventory (shard headers only, no weight download)…")
    rows = hub.fetch_all_headers(
        repo, progress=lambda s: print(s, end="\r", file=sys.stderr))
    print(file=sys.stderr)

    total_bytes = sum(census.nbytes(r) for r in rows)
    params = census.param_count(rows)
    buckets, unmatched = census.classify(rows)

    print(f"tensors:           {len(rows)}   [MEASURED from shard headers]")
    print(f"parameters:        {params / 1e12:.2f}T ({params:,})   [CALCULATED from shapes]")
    print(f"weight download:   {fmt_bytes(total_bytes)}   [MEASURED from offsets]")
    import collections
    dt = collections.Counter(r["dtype"] for r in rows)
    print(f"dtypes:            {dict(dt)}")
    fam = "  ".join(f"{b}:{len(buckets[b])}" for b, _ in census.BUCKETS if buckets[b])
    print(f"tensor families:   {fam}")
    if unmatched:
        print(f"⚠ unclassified:    {len(unmatched)} tensors (first: {unmatched[0]['name']})")

    ram = local_ram_bytes()
    disk_free = shutil.disk_usage(".").free
    gpus = local_gpus()
    print("\nyour machine   [MEASURED]")
    print(f"  host:  {platform.system()} {platform.machine()}")
    print(f"  RAM:   {fmt_bytes(ram) if ram else 'unknown'}")
    print(f"  disk:  {fmt_bytes(disk_free)} free")
    print(f"  GPUs:  {'; '.join(gpus) if gpus else 'none detected (nvidia-smi)'}")

    print("\nfeasibility   [CALCULATED: weight bytes vs local resources; runtime overhead not included]")
    ok_disk = disk_free > total_bytes * 1.05
    print(f"  {'✓' if ok_disk else '✗'} disk for download: need {fmt_bytes(total_bytes)}, have {fmt_bytes(disk_free)}")
    if ram:
        ok_ram = ram > total_bytes
        print(f"  {'✓' if ok_ram else '✗'} full weights in RAM: need {fmt_bytes(total_bytes)}, have {fmt_bytes(ram)}")
        if not ok_ram and n_experts and topk:
            active_frac = (topk + (hp("num_shared_experts", "n_shared_experts", default=0) or 0)) / n_experts
            moe_bytes = sum(census.nbytes(r) for r in buckets["moe"])
            dense_bytes = total_bytes - moe_bytes
            hot = dense_bytes + moe_bytes * active_frac
            print(f"  · per-token active weights: ~{fmt_bytes(int(hot))} "
                  f"({topk}/{n_experts} experts) — a LOWER BOUND on the working set, "
                  f"not a residency verdict: the cumulative set across a real "
                  f"generation depends on routing (measure it; do not assume it)")
    print("\nspeed/cost estimates: not shown — theseus reports measured numbers only.")
    return 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: theseus inspect <org/repo>")
        return 2
    return run(args[0])
