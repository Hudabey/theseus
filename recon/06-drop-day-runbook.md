# 06 — Drop-day runbook (K3 weights release, announced for July 27)

Ordered hour-zero checklist. Principle: **small files before big ones** — modeling code
and config define everything; the 2.8T weights come last, and most decisions need zero
weight bytes. Every step lists the exact command, what to extract, and which recon
01/04/05 open question (OQ) it closes. Decision tree in §4.

Example environment: any Python ≥ 3.9 with `huggingface_hub`, `safetensors`,
`requests`, and `numpy`; the `hf` CLI and `jq` on PATH. `$PY` below is that
interpreter — verify the imports before drop day, not during it. Exact rehearsal
toolchain (2026-07-22): Python 3.12.2, `huggingface_hub` 1.3.5 (also provides the
`hf` CLI), `safetensors` 0.5.2, `requests` 2.32.4, `numpy` 1.26.4, `jq` 1.6.

The helper scripts invoked below as `tools/drop_day/*.py` are five small standalone
HF-API utilities (release watcher, completeness gate, shard-header fetch, tensor-name
classifier, small-tensor puller; ~100 lines each — included in this repo under
[`tools/drop_day/`](../tools/drop_day/); `pull_small.py --self-test` is the offline
BF16-decode unit test referenced in Step 3). They were
**rehearsed end-to-end on 2026-07-22 against
`moonshotai/Kimi-Linear-48B-A3B-Instruct`** (same lineage): completeness gate
passed 20/20 shards, header fetch returned 20,493 tensors with zero shard downloads,
classification bucketed all of them with **0 unmatched** (so the bucket regexes match
Moonshot's real HF naming — `self_attn.A_log`, `block_sparse_moe.experts.N.w1`, …),
and BF16/F32 range-request pulls printed correct non-zero norms
(`input_layernorm.weight` BF16 → norm 9.70; `A_log` F32 `[1,1,32,1]` → 15.77).

```bash
export REPO=moonshotai/Kimi-K3       # ← placeholder; fix to the real repo id first thing
export PY=python3                    # a python with huggingface_hub/safetensors/requests/numpy
export WORK="${XDG_CACHE_HOME:-$HOME/.cache}/theseus-k3-drop"  # outside any repo: pulled
                                     # files must never be committable by accident
mkdir -p $WORK && cd $WORK
# if gated: export HF_TOKEN=...
```

One caveat baked into this plan: **the safetensors index JSON lists tensor *names*
only** (`weight_map`: name → shard file). Shapes and dtypes live in each shard's
header, which is fetchable via HTTP range requests without downloading weights — step
2 handles this.

---

## Step 0 — findings artifact + drop detection (before T+0)

### 0a. The findings artifact — every step writes here, immediately

Create `drop/FINDINGS.md` **before** the release from a fixed template: one row per
open question and decision (01-OQ1, 04-OQ2/3/7/8, 05-OQ1–5, D1–D7), columns
*question / finding / evidence (file:line or tensor name) / decision taken*, plus a
step log and a Step-1d template-diff section. **The rule: a step is not done
when its command exits — it is done when its row in FINDINGS.md is filled in.**
Terminal output is not a record; the table is. Commit FINDINGS.md after each step
completes so the timeline is in git.

### 0b. Drop detection

```bash
$PY tools/drop_day/watch_release.py --interval 180
# polls the moonshotai HF org. Alerts on ANY of: a new repo id matching /k3/i, a
# revision-SHA change on a matching repo, config.json appearing, or
# model.safetensors.index.json appearing — a repo created empty (or flipped
# private→public) and populated later must still trigger. Honors HF_TOKEN.
# Rings the bell and prints `export REPO=...`. Baseline 2026-07-22: 18 repos, no match.
```

### 0c. Completeness gate — do NOT start Step 1 against a partial upload

Frontier releases upload hundreds of shards; the index often lands before the last
shard. First command once the repo exists:

```bash
$PY tools/drop_day/check_complete.py $REPO
# verifies model.safetensors.index.json exists and every shard it names is on the hub
# (and reports hub-side .safetensors files the index doesn't name, e.g. vision).
# Exit 0 = proceed to Step 1. Exit 1 = wait and re-run.
```

---

## Step 1 — config.json + modeling code (T+0, no weights)

### 1a. Pull every small file

```bash
hf download $REPO --include "*.json" "*.py" "*.jinja" "*.txt" "*.model" \
    --exclude "*.safetensors" --local-dir $WORK/repo
ls -la $WORK/repo
```

This gets `config.json`, any `modeling_*.py` / `configuration_*.py` (K2 shipped
trust-remote-code modeling files; if K3 doesn't, the transformers PR is the modeling
source — find it via `grep -r "kimi_k3\|kimik3" in the transformers repo` or the vLLM
release branch, which recon 05 confirms has "model definitions … integrated"),
tokenizer files, chat template, and `model.safetensors.index.json`.

### 1b. Config extraction checklist

```bash
jq . $WORK/repo/config.json | tee $WORK/config-pretty.json
```

| extract | keys to look for | closes |
|---|---|---|
| **SiTU** | `hidden_act` / `moe_act` / anything ≠ `silu`; `quantization_config`; then grep modeling code (1c) for the formula | 05-OQ1 |
| **MoE structure** | `n_routed_experts` (=896?), `num_experts_per_tok` (=16?), `n_shared_experts`, `moe_intermediate_size`, anything named `latent`/`stable` | 05-OQ2 |
| **AttnRes** | `attnres_block_size` (fla name, [`configuration_kda.py:48`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/models/kda/configuration_kda.py#L48)) or any `*res*`/`*residual*` key; value `1` = full mode, even N = block mode (recon 04 §3, fla semantics) | 04-OQ2, 05-OQ4 |
| **Layer semantics** | `num_hidden_layers` (=93? decoder-only?), `layer_types` list vs `full_attention_interval`-style key, MLA placement offset (93 mod 4 ≠ 0 — recon 05 A12) | 01-OQ4, 05-OQ3 |
| **MLA gate** | MLA keys (`q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`…) + any `attn_gate`/`gate_proj`-adjacent key on the attention config | 05-OQ5 |
| **KDA numerics vs Kimi-Linear** | `head_dim` (128?), `num_heads`, `num_v_heads` (GVA?), `conv_size` (4?), `allow_neg_eigval`, `safe_gate`/`lower_bound`, chunk size if exposed | 01-OQ1 — gates how much delta-net kernel work remains |
| **All norm eps** | `rms_norm_eps` / `norm_eps` — recon 04 assumes one shared eps for AttnRes key-norm and folded norm (`modeling_kda.py:91,140`) | 04-OQ8 |
| **Vocab/context** | `vocab_size`, `max_position_embeddings` (1M?), `rope_theta` for MLA layers | GGUF metadata |

### 1c. Modeling-code extraction (the SiTU + LatentMoE + AttnRes ground truth)

```bash
grep -rn -i "situ" $WORK/repo/*.py                     # formula + parameter tensors
grep -rn -i "latent\|shared_expert\|class .*MoE" $WORK/repo/*.py
grep -rn -i "res_proj\|res_norm\|attnres\|residual" $WORK/repo/*.py
grep -rn -i "gate" $WORK/repo/*.py | grep -i "attn\|mla"
grep -rn "eps" $WORK/repo/*.py | grep -i "norm"
```

Read the hits in full, then answer, in order:

1. **SiTU forward math**, verbatim, and whether its parameters are per-expert tensors,
   per-layer tensors, or config scalars (05 §3 established only that "SiTU parameters"
   exist). → decision D1.
2. **Stable LatentMoE class structure**: is routing standard
   `softmax/sigmoid(router(x)) → top-k → experts + shared_expert add` (i.e. branding),
   or is there a latent projection / factored expert structure? → D2.
3. **AttnRes wiring**: diff against fla's `modeling_kda.py:107-185` (recon 04 §2) —
   prefix-sum discipline, boundary arithmetic, folded output norm, and **whether the
   top-level aggregation before the head exists**. → D3.
4. **MLA gate wiring**: where `sigmoid ⊙ multiply` attaches (recon 05 A6) — on attn
   output vs inside the latent path. → D4.

Record every answer in `drop/FINDINGS.md` with the modeling-code file:line as
evidence before moving on.

### 1d. Tokenizer + chat template (do this before any tensor work)

Rationale: Unsloth's documented DeepSeek-V4 release experience — multi-turn
conversations misbehaved against the HF baseline across GGUF providers, and the fix
was a chat-template rework ("tested over 4000 conversations"), not a
tensor-conversion one (https://unsloth.ai/docs/models/deepseek-v4). The template is a
deliverable, not an afterthought.

```bash
# K3 template + tokenizer (already in $WORK/repo from 1a; if not:)
hf download $REPO --include "tokenizer*" "*.jinja" --local-dir $WORK/repo
# K2.5 baseline for the diff:
hf download moonshotai/Kimi-K2.5 --include "tokenizer_config.json" "*.jinja" \
    --local-dir $WORK/k25
jq -r '.chat_template // empty' $WORK/repo/tokenizer_config.json > $WORK/k3-template.jinja \
  || cp $WORK/repo/chat_template.jinja $WORK/k3-template.jinja
jq -r '.chat_template // empty' $WORK/k25/tokenizer_config.json > $WORK/k25-template.jinja
diff -u $WORK/k25-template.jinja $WORK/k3-template.jinja | tee $WORK/template.diff
```

Record in FINDINGS.md (evidence = template line numbers):

- **Reasoning-mode markers**: any `<think>`-style tags, `reasoning`/`thinking` role or
  flag handling, and whether reasoning content is dropped from history on re-render.
- **Tool-call format changes** vs K2.5 (vLLM lists "tool calls, reasoning output,
  structured-output paths" as K3 serving semantics — recon 05 §4): section tokens,
  JSON vs token-delimited arguments, parallel-call framing.
- **New special tokens**: diff `added_tokens` / `special_tokens_map` between the two
  repos; note ids — these go into GGUF metadata verbatim.
- **Template control flow we must reproduce** in the GGUF-embedded template: loops
  over tools, system-prompt defaulting, vision placeholders (image-only per recon 05
  A3), role alternation constraints.
- Sanity: `vocab_size` in config.json (1b) vs tokenizer's actual vocab length.

---

## Step 2 — full tensor inventory, still no weight download (T+15)

### 2a. Names + shapes + dtypes from shard headers

```bash
cd $WORK && $PY tools/drop_day/fetch_headers.py $REPO -o tensors.json
```

Fetches only each shard's header (8-byte length + JSON) via HTTP Range requests and
writes `tensors.json` with name/dtype/shape/shard/**data_offsets** — the offsets are
what step 3 uses for single-tensor pulls. Rehearsed: 20 shards / 20,493 tensors from
Kimi-Linear in under a minute, zero weight bytes. (Alternative if range requests
misbehave: `huggingface_hub.get_safetensors_metadata`, which lacks offsets — then
step 3 needs one small shard downloaded instead.)

### 2b. Classify every name; the *unmatched bucket is the discovery list*

```bash
$PY tools/drop_day/classify_tensors.py tensors.json
```

Buckets (first match wins): `attnres` (recon 04 §4 names), `kda`
([`conversion/kimi_linear.py:158-179`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/conversion/kimi_linear.py#L158-L179) names), `mla`, `moe`, `mlp`, `norms`, `embed`,
`vision`; everything else lands in `unmatched.json`. The patterns are validated
against real Moonshot naming (the Kimi-Linear rehearsal bucketed all 20,493 tensors
with zero unmatched), but treat `unmatched.json` as the **first discovery queue, not
proof of new architecture**: renamed ordinary tensors land there too, and genuinely
new tensors can be swallowed by the broad buckets (`gate`, `norm`, `experts` match a
lot). After triaging unmatched, **audit every matched bucket** for unexpected names,
per-layer counts, shapes, and dtypes before declaring the inventory understood. Copy
the unmatched list and any bucket anomalies into FINDINGS.md as evidence rows.

Then targeted checks:

- **AttnRes granularity** (04-OQ3): count `res_proj`-like tensors for one layer —
  **2 per layer** (`attn_res_proj` + `mlp_res_proj`, shapes `[1, D]`) = fla
  per-sublayer layout (recon 04 §4); **1** = per-layer variant (converter diverges);
  **0** = different naming → back to 1c hits. Also: does `model.res_proj.weight`
  exist → top-level aggregation confirmed (04-OQ2).
- **SiTU tensors** (05-OQ1): whatever the modeling code named in 1c — record name
  pattern, per-expert vs per-layer count, shapes, dtype. If nothing matches, SiTU
  params are config scalars → best case.
- **MXFP4 serialization** (01-OQ5): which tensors are quantized (expect routed-expert
  weights only), and the packing convention — paired `*_blocks` (U8, last dim = row/2
  nibbles) + `*_scales` (U8/E8M0, last dim = row/32) is the GPT-OSS-style layout that
  maps 1:1 onto ggml `block_mxfp4` (`QK_MXFP4=32`, `e:uint8 + qs[16]`); a single
  fused tensor or different group size triggers D6. Also confirm what stays
  high-precision: norms, AttnRes tensors, KDA conv (mainline forces F32 —
  recon 01 §4), embeddings, MLA.
- **Layer census** (05-OQ3): `jq` over names — max `layers.N` index +1 vs
  `num_hidden_layers` vs "93"; which layer indices have MLA tensor names vs KDA ones →
  exact placement/offset, settles 01-OQ4's signaling design.

```bash
jq -r '.[].name' tensors.json | grep -oE 'layers\.[0-9]+' | sort -u -t. -k2 -n | tail -1
for i in $(seq 0 12); do
  echo -n "layer $i: "; jq -r '.[].name' tensors.json | grep "layers\.$i\." | grep -q "kv_a_proj\|q_a_proj" && echo MLA || echo KDA
done
```

### 2c. Diff against the converter's expected mapping

Expected-name baseline = Kimi-Linear converter
(`vendor/llama.cpp/conversion/kimi_linear.py:158-179`, recon 01 §5 items #7/#9) +
recon 04 §4 AttnRes table. Anything in `unmatched.json` after 2b is either SiTU,
LatentMoE structure, the MLA gate, or something nobody predicted — each goes through
§4 before any converter code is written.

---

## Step 3 — small-weight pulls only (T+45; a few MB, not the 2.8T)

Single-tensor fetch via the `data_offsets` recorded in `tensors.json`
(`tools/drop_day/pull_small.py`). Dtype handling is exact — **BF16 is decoded as
`uint16 → uint32 << 16 → view(float32)`** (bfloat16 is the top half of an IEEE
float32); F16/F32 cast to f32; packed U8/FP8 quant bytes are reported min/max without
a bogus float view. The decoder has an offline unit test with a known BF16 tensor
(run it once on drop day before trusting any norm):

```bash
$PY tools/drop_day/pull_small.py --self-test
# (a) AttnRes pseudo-query liveness — recon 04 OQ7
$PY tools/drop_day/pull_small.py $REPO --tensors tensors.json --match 'res_proj'
# (b) one expert's quant tensors for repacker verification — recon 01 item #8
$PY tools/drop_day/pull_small.py $REPO --tensors tensors.json \
    --match 'experts\.0\..*' --limit 4 --save
```

Rehearsed against Kimi-Linear: BF16 `input_layernorm.weight [2304]` → norm 9.70,
F32 `A_log [1,1,32,1]` → norm 15.77 — non-zero floats, correct shapes.

- **(a) Pseudo-query norms** (04-OQ7): non-zero ⇒ mechanism trained/live. All-zero ⇒
  either the mechanism is inactive (uniform softmax = depth-*mean*, still not a no-op)
  or we're reading the wrong tensors — stop and re-check naming before building.
- **(b) One expert block**: dequantize HF-side (`(blocks nibbles → e2m1 values) ×
  2^(scales−127)` per 32-group) and verify a hand-repack into ggml `block_mxfp4`
  round-trips bit-exact vs `ggml_dequantize_row_mxfp4`. This is the **go/no-go for
  recon 01 item #8** (native MXFP4 passthrough) before touching 2.8T of data.
- Optional third pull: one full KDA layer's small tensors (`A_log`, `dt_bias`,
  `conv1d`) to spot-check shapes against `conversion/kimi_linear.py:158-179`
  assumptions (01-OQ1).

Write the norms, the repack verdict, and the saved-file names into FINDINGS.md
before opening the decision tree.

---

## Step 4 — decision tree

Each finding → what it unblocks / which contingency fires.

**D1 — SiTU** (from 1c formula + 2b tensors; closes 05-OQ1)
- Composes from existing ggml ops (e.g. a parameterized sigmoid/tanh family —
  `ggml_sigmoid`, `ggml_tanh`, `mul`, `add`, `scale` all exist) → **graph
  composition**; converter emits its params (tensors or `hparams` metadata); MoE path
  unblocked at cost S.
- Per-expert *tensor* parameters → new GGUF tensor type entries + they must survive
  next to MXFP4 expert weights; cost M, converter + arch enums.
- Genuinely new math (piecewise, LUT, anything without a ggml primitive) → **new op +
  per-backend kernels**; contingency: CPU-only op first, CUDA next, publish the gap.
  Timeline slip; announce in the repo immediately.

**D2 — Stable LatentMoE** (1c class structure; closes 05-OQ2)
- Standard shared+routed (branding) → reuse existing MoE graph machinery; only counts
  (896/16/shared) go into GGUF metadata. Unblocks converter + graph, cost S.
- Latent/factored routing (extra down-projection before router, or shared expert
  factors) → new graph wiring + new tensors; cost M; kimi-linear graph no longer a
  pure clone.
- Structurally different expert weights → re-scope recon 01 item #3 entirely; L.

**D3 — AttnRes config** (1b/2b; closes 04-OQ2/3, 05-OQ4)
- Even `attnres_block_size`, fla-style tensors, top-level aggregation present → recon
  04 §5 graph composition ships day-one; source-count small; unblocks the kimi-k3
  graph work.
- Full mode (`1`) with 93 layers → 187 sources at the head, O(layers²) nodes →
  **contingency: fused AttnRes op required before an e2e run**; model it on
  `fla/ops/attnres/fused.py:38-121`.
- Config key absent / wiring diverges from fla → recon 04 is reference-only; re-derive
  from K3 modeling code before writing any graph code; re-run recon 04 §7-style
  equation check.
- Pseudo-queries all zero (3a) → double-check tensor mapping first; if genuinely zero,
  implement anyway (depth-mean is not identity) but flag as a likely mapping bug.

**D4 — MLA gate** (1c; closes 05-OQ5)
- Gate on attention output (`o = σ(W_g x) ⊙ attn_out`) → compose with `mul_mat` +
  `ggml_sigmoid` + `mul`; S. (Same shape as KDA's output gate — precedent exists.)
- Gate inside the latent/rope path → modify the MLA build in the kimi-k3 graph clone;
  M.

**D5 — layout signaling** (1b/2b layer census; closes 01-OQ4, 05-OQ3)
- Config has explicit `layer_types` per layer → emit as per-layer metadata; map to
  `is_recr` like Kimi-Linear's `n_head_kv==0` trick or an explicit list; S.
- Interval-only key → decide offset from the 2b census (93 mod 4 ambiguity resolved
  empirically); may need a new GGUF KV key; S.

**D6 — MXFP4 packing** (2b dtypes + 3b round-trip; closes 01-OQ5)
- blocks/scales, 32-group, e2m1 nibbles + E8M0 scale → **passthrough repack** (pure
  byte shuffle, no requant); unblocks recon 01 item #8 as M-not-L.
- Different group size / scale format / fused layout → translation step with a
  dequant-requant *only if lossless repack is impossible*; document any lossy step
  loudly; L, and it gates everything downstream of the converter.

**D7 — KDA numerics** (1b config vs Kimi-Linear; closes 01-OQ1)
- Identical (head_dim 128, conv 4, sigmoid gates, `-exp(A_log)`) → existing kernels
  **may be reusable with little or no recurrence-kernel work**, `ggml_gated_delta_net`
  reused if K3 matches the supported shapes and conventions (recon 01 §4).
- Any divergence (GVA `num_v_heads≠num_heads`, different chunk/conv, safe_gate) →
  check the existing op's parameter surface first (`ggml.h:2569`) — most variants are
  flags, not new kernels; escalate only if the recurrence itself changed.

---

## Open questions

1. **The repo id and distribution channel** — `moonshotai/Kimi-K3` is a guess;
   confirm on announcement (HF vs Moonshot's own hub; gated or not). Every command
   above keys off `$REPO`.
2. **Does K3 ship trust-remote-code modeling files** (K2 did) or land as a
   transformers/vLLM PR only? Step 1c's ground truth moves accordingly — vLLM's
   release branch is the fallback source (recon 05 §4).
3. **Is the vision tower in the same checkpoint index** (affects 2b bucket noise and
   the layer census), and is "93" decoder-only (05-OQ3)?
4. **Does safetensors carry MXFP4 as U8 blocks/scales pairs** (assumed, GPT-OSS
   precedent) or as a new dtype string? The 2a dump answers this in seconds; D6
   handles the branches.
5. ~~Will HF range-requests work against Xet-backed storage?~~ **Answered in
   rehearsal (2026-07-22): yes** — header fetch and single-tensor pulls both worked
   against `moonshotai/Kimi-Linear-48B-A3B-Instruct`. Residual risk only if K3's repo
   is served differently; fallback stands: `hf download` one smallest shard and read
   headers locally (`safetensors.safe_open`).
6. **KDA chunk size is not usually in config.json** — if absent, 01-OQ1's chunk-16
   confirmation needs the modeling/kernel code (1c), not the config.
