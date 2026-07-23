# 02 — Native MXFP4 preservation: HF → GGUF without re-quantization

Scope: the exact ggml `block_mxfp4` on-disk/in-memory format, the two shipping
converter precedents that already preserve source MXFP4 into GGUF (**gpt-oss** and
**DeepSeek-V4**), where higher-precision tensors are enforced, and the checklist a K3
repacker must satisfy for the conversion to be **lossless at the byte level**. This is
recon 01 §4/§5 **item #8** (difficulty L) and closes what is closable of **01-OQ5**
before the weights drop; recon 05 **A11** confirms MXFP4 is K3's release format. Every
claim carries a file:line ref against `vendor/llama.cpp` @ `1a064ab` (2026-07-22);
permalink base:
[github.com/ggml-org/llama.cpp @ 1a064ab](https://github.com/ggml-org/llama.cpp/tree/1a064ab0921238c1daa397d6f4a900ef33884de2).

> **Anchor.** "Native MXFP4 → GGUF" is **not** a quantization step. It is a pure byte
> permutation (nibble reorder + scale-byte relocation). The lossy encoder
> (`quantize_row_mxfp4_ref`) exists in the same tree and must never run on
> already-MXFP4 tensors — its scale-selection rule has no reason to reproduce the
> checkpoint's stored scales (§1.4). Both shipping precedents route around it.

---

## TL;DR

- ggml's `block_mxfp4` = **17 bytes per 32 elements**: 1 scale byte (raw E8M0) + 16
  nibble bytes in a **split-half** convention — `qs[j]` holds element `j` (low nibble)
  and element `j+16` (high nibble) (§1.2).
- HF safetensors precedents store an **adjacent-pair** convention — byte `k` holds
  element `2k` (low) and `2k+1` (high) — plus a separate raw-E8M0 scales tensor
  (§2.1). Conversion = re-pairing nibbles + prepending the scale byte. Nothing is
  recomputed; scale bytes pass through **unchanged**.
- The two dequant conventions differ by an exact factor-of-2 split — ggml stores
  E2M1 values **doubled** and a **halved** scale decode — so HF `v·2^(e−127)` and ggml
  `(2v)·2^(e−127)/2` produce **bit-identical float32**, including subnormal-scale
  edge cases (§1.3).
- In the converter, `MOSTLY_MXFP4_MOE` is a **file-level label applied after the
  tensors are written**, not a per-tensor selection rule; per-tensor MXFP4 selection
  by rule exists only in the separate requantize tool (§3.2).
- Our repacker (`src/repack.py`) already implements the full mapping and
  is proven byte-identical to the gpt-oss converter's transform on random input
  (§5); its oracle suite (`tests/test_repack_mxfp4.py`) is the pre-drop
  acceptance gate, extended by the on-checkpoint test defined in §6.

---

## 1. ggml ground truth: `block_mxfp4`

### 1.1 The struct

[`ggml/src/ggml-common.h:214-219`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/ggml/src/ggml-common.h#L214-L219):

```c
#define QK_MXFP4 32
typedef struct {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");
```

17 bytes per 32 elements. Registered as `GGML_TYPE_MXFP4 = 39`
(`ggml/include/ggml.h:429`), type traits `blck_size = 32`, `type_size = 17`,
`to_float = dequantize_row_mxfp4`, `from_float_ref = quantize_row_mxfp4_ref`
(`ggml/src/ggml.c:751-758`). gguf-py mirrors the sizes:
`GGML_QUANT_SIZES[MXFP4] = (32, 1 + 16)` (`gguf-py/gguf/constants.py:4775`),
`GGMLQuantizationType.MXFP4 = 39` (`constants.py:4593`).

Blocks run along **ne[0]** — the ggml row axis, i.e. the matmul contraction axis. The
GGUF reader hard-rejects any tensor whose `ne[0]` is not a multiple of the block size
(`ggml/src/gguf.cpp:712-718`).

### 1.2 Nibble convention — which element each nibble holds

The decode loop is the ground truth
([`ggml/src/ggml-quants.c:569-586`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/ggml/src/ggml-quants.c#L569-L586)):

```c
const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];
y[i*qk + j + 0   ] = x0*d;
y[i*qk + j + qk/2] = x1*d;
```

Stated explicitly, for `j = 0..15` within a 32-element block:

| byte | low nibble (`& 0x0F`) | high nibble (`>> 4`) |
|---|---|---|
| `qs[j]` | element `j` | element `j + 16` |

This is a **split-half** layout: the block's first 16 elements live in the low
nibbles of `qs[0..15]`, the second 16 in the high nibbles. The dot-product kernels
depend on exactly this order (`ggml/src/ggml-cpu/quants.c:320-322` pairs
`y.qs[j]` with `x.qs[j] & 0xf` and `y.qs[j+16]` with `x.qs[j] >> 4`), so a wrong
permutation produces a *loadable* model that multiplies activations against permuted
weights — it fails numerically, not loudly.

The 4-bit code is an index into the shared FP4 value table
(`ggml/src/ggml-common.h:1124-1129`):

```c
// e2m1 values (doubled), shared by MXFP4 and NVFP4
GGML_TABLE_BEGIN(int8_t, kvalues_fp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
GGML_TABLE_END()
#define kvalues_mxfp4 kvalues_fp4
```

Index = the raw OCP E2M1 code (bit 3 = sign): codes 0–7 → doubled values
{0,1,2,3,4,6,8,12}, codes 8–15 → their negatives. Code 8 is OCP's **−0** and decodes
to integer `0` (the sign of zero is dropped at decode — relevant for §6).

### 1.3 Scale convention — and why passthrough is bit-exact

`d = GGML_E8M0_TO_FP32_HALF(e)`, i.e. `2^(e−127)/2`
(`ggml/src/ggml-impl.h:477-495`), with an explicit denormal branch for `e < 2`
(`bits = 0x00200000 << e`, `ggml-impl.h:481-486`) and — per the comment at
`ggml-impl.h:490` — **no NaN handling** (E8M0 `0xFF` is NaN in the OCP spec).

The OCP/HF-side semantics are `value = e2m1(code) · 2^(e−127)` with un-doubled E2M1
elements {0, ±0.5, …, ±6}. ggml computes `(2·e2m1(code)) · (2^(e−127)/2)`. The two
formulations are algebraically equivalent through exact power-of-two scaling — this
covers the `e ∈ {0,1}` subnormal-scale cases (the halved decode keeps them
representable) and the overflow edge (`e ≥ 253` with |element| ≥ 8 → `+inf`/`−inf` on
both sides) — and the oracle tests verify float32 **bit equality** across normal,
subnormal, overflow, and signed-zero code cases
(`tests/test_repack_mxfp4.py`). Consequence: **the scale byte is copied, never
recomputed** — both shipping converters do exactly that (§2).

Caveat on evidence: the vendor tree contains **no HF-side MXFP4 dequant** to cite.
The source convention is stated directly in the DeepSeek-V4 converter
(`conversion/deepseek.py:617-618`) and independently encoded by the gpt-oss
converter's repacking transform (`conversion/gpt_oss.py:23-46` — the two are proven
byte-equivalent by `test_gptoss_transform_equivalence`, §5). Our repacker carries its
own HF-side reference decoder for this reason (§5).

### 1.4 The lossy path a repacker must avoid

`quantize_row_mxfp4_ref` (`ggml/src/ggml-quants.c:350-382`) chooses
`e = floor(log2(amax)) − 2 + 127` and nearest-value codes via `best_index_mxfp4`
(`ggml-quants.c:337-348`); gguf-py's `MXFP4.quantize_blocks` mirrors it
(`gguf-py/gguf/quants.py:669-690`, scale rule at `:675`, `argmin` at `:682`). Two
reasons dequant→requant is **not** lossless even when the values are exactly
representable:

1. The scale rule need not reproduce the checkpoint's stored `e` (training-side
   quantizers may pick saturation-aware or rounded-up exponents); a different `e`
   changes every nibble in the block.
2. `argmin`/first-hit tie-breaking canonicalizes codes: −0 (code 8) becomes +0
   (code 0), and exact midpoints resolve by table order. Bytes change even where
   floats don't.

Byte-level passthrough sidesteps both. This is the precise content of runbook **D6**'s
"passthrough repack (pure byte shuffle, no requant)" branch
(`recon/06-drop-day-runbook.md` §Step 4).

---

## 2. The shipping precedents — what "native MXFP4" required in practice

### 2.1 gpt-oss (`conversion/gpt_oss.py`)

**Detection & dequant bypass.** `quantization_config.quant_method == "mxfp4"` sets
`self._is_mxfp4` (`conversion/base.py:791-827`, assignment at `:827`);
`GptOssModel.dequant_model` returns early so the generic dequantizer never touches the
expert tensors ([`conversion/gpt_oss.py:17-21`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/conversion/gpt_oss.py#L17-L21) — annotated "TODO: remove once MXFP4
is supported more generally"; MXFP4 passthrough is a per-model opt-in, not
infrastructure).

**Source layout.** Paired uint8 tensors: `*_blocks` `[n_exp, rows, n_blocks, 16]` and
`*_scales` `[n_exp, rows, n_blocks]` (asserted 4-D, last dim 16 —
`gpt_oss.py:24-25,52-53`), adjacent-pair nibbles, raw E8M0 scales.

**The repack** (`gpt_oss.py:23-61`): `transform_nibble_layout` (`:23-46`) is a chain
of torch nibble shuffles; `repack_mxfp4` (`:48-61`) prepends the scale byte via
`torch.concat((scales, blocks), dim=-1)` (`:55`) — **scale bytes pass through
unchanged** — and writes with
`add_tensor(..., raw_dtype=gguf.GGMLQuantizationType.MXFP4)` (`:61`). Despite its
opacity, the transform is exactly the adjacent-pair → split-half permutation of §1.2:
a numpy port of `gpt_oss.py:23-46` plus the `:55` concat is **byte-identical to our
repacker** on random `[64, 90, 16]` input (verified 2026-07-23, §5).

**Fused-tensor handling at row granularity.** HF fuses gate/up with **row-interleaved**
experts: the converter splits blocks `[:, ::2, :, :]` / `[:, 1::2, :, :]` and scales
likewise (`gpt_oss.py:73-80`). The split axis is the *row* axis — whole quantized rows
move; no 32-block is ever cut. Biases follow the same interleave on their own axis
(`:106-112`).

**No transpose in the MXFP4 path.** The bf16 *fallback* path transposes
(`gpt_oss.py:100,117`) — the MXFP4 path does not, i.e. the HF blocks already have the
contraction dimension as the quantized last axis. A source layout that grouped along
the other axis could not be passed through (checklist §4-1).

**Shape derivation from bytes.** The writer maps byte shape → logical shape when
`raw_dtype` is given and the array is uint8
(`gguf-py/gguf/gguf_writer.py:358-360` → `quant_shape_from_byte_shape`,
`gguf-py/gguf/quants.py:21-26`): last dim must divide by 17, logical last dim =
`bytes/17·32`.

**Ordering fragility.** Blocks are buffered in a local until the matching scales
tensor arrives — "we assume that tensors are loaded in the correct order"
(`gpt_oss.py:63-81`). Works for gpt-oss's shard layout; not a pattern to copy (K3:
pair by name, as DSv4 does).

### 2.2 DeepSeek-V4 (`conversion/deepseek.py`) — the second, cleaner precedent

`DeepseekV4Model` (`deepseek.py:474+`) stores experts **per-expert** as
`….{eid}.{proj}.weight` (packed u8 `[out_features, cols/2]`) + `….scale`
(`[out_features, cols/32]`), fetched by name — no ordering assumption
(`deepseek.py:628-641`). `_pack_mxfp4_blocks` (`:600-622`) states the convention in
comments — "safetensors bytes store adjacent values as low/high nibbles … ggml MXFP4
blocks store values 0..15 in low nibbles and 16..31 in high nibbles" (`:617-618`) —
and builds `qs = vals[:,:,:16] | (vals[:,:,16:] << 4)` with the scale byte
concatenated first (`:619-622`): structurally identical to our repacker. Experts are
stacked into `[n_experts, rows, n_blocks·17]`, the logical shape computed explicitly,
and written raw (`:643-647`). Triggered unconditionally for w1/w2/w3 of every layer in
`generate_extra_tensors` (`:668-682`).

Notable differences from gpt-oss: source dtypes are recorded per tensor and
**preserved** (`_collect_source_dtypes`, `deepseek.py:508-514`; `tensor_force_quant`
returns BF16/F32/Q8_0-for-dequantized-FP8 accordingly, `:761-771`), and the ftype is
forced to `MOSTLY_MXFP4_MOE` right after `prepare_tensors()` (`:773-776`).

### 2.3 What "native MXFP4" required, distilled

1. detect the quant method + **bypass the generic dequantizer** for the packed pair;
2. a **pure nibble permutation** (adjacent-pair → split-half), no arithmetic;
3. **scale byte passthrough** as block byte 0;
4. de-fuse/split fused projections at **whole-row granularity** only;
5. write with `raw_dtype=MXFP4`, shape derived from bytes (`/17·32`);
6. **no transpose** — source must already be contraction-dim-last;
7. stack experts to 3-D (or keep the checkpoint's stacking);
8. everything not repacked continues down the normal precision path (§3).

---

## 3. Which tensors stay higher-precision, and where that's enforced

### 3.1 Converter side (`conversion/base.py`, `prepare_tensors` loop)

Applied to every tensor that is *not* written raw:

| Rule | Effect | Where |
|---|---|---|
| `n_dims <= 1 or new_name.endswith("_norm.weight")` | **F32** | `base.py:892-893` — all norms, biases, per-head scalars (`A_log`, `dt_bias`, attn sinks) |
| Always-F32 tensor list | **F32** | `base.py:897-931` — includes `FFN_GATE_INP` (**router**, `:901`) and `FFN_GATE_INP_SHEXP`, `SSM_CONV1D`, and the **Kimi KDA convs** `SSM_CONV1D_Q/K/V` ("Kimi KDA conv weights should be F32", `:919-922`), `INDEXER_PROJ` |
| Name not ending `.weight/.lora_a/.lora_b` | **F32** | `base.py:927-929` |
| Everything else | session ftype | `base.py:950-964` — first-tensor dtype heuristic picks BF16/F16 (`base.py:161-176`), so gpt-oss/K3-style bf16 attention + **embeddings + output** land as BF16 |

gpt-oss's router maps to `FFN_GATE_INP` via `"model.layers.{bid}.mlp.router"`
(`gguf-py/gguf/tensor_mapping.py:456`) → F32 by the list rule. DSv4 instead pins each
tensor to its **source** dtype (`deepseek.py:761-771`) — the safer template when a
checkpoint mixes f32/bf16 deliberately.

### 3.2 `MOSTLY_MXFP4_MOE` is two different things

- **Converter:** a *label*. `write()` runs `prepare_tensors()` **before**
  `prepare_metadata()` (`base.py:1024-1026`); the promotion to
  `MOSTLY_MXFP4_MOE` (`base.py:1000-1004`) happens in metadata prep, after every
  tensor is already written. It is not in the per-tensor ftype dispatch
  (`base.py:950-964` — reaching that chain with `MOSTLY_MXFP4_MOE` set would raise
  `Unknown file type`, `:964`), which is also why DSv4 defers forcing it until after
  `super().prepare_tensors()` (`deepseek.py:773-776`).
- **Requantize tool (`llama-quantize`, from an existing f16/bf16 GGUF):** a real
  per-tensor rule — `ne[2] > 1` (stacked-expert 3-D) → `GGML_TYPE_MXFP4`, every other
  2-D tensor → `Q8_0` (`src/llama-quant.cpp:465-472`), `output`/`token_embd` → `Q8_0`
  (`:450-452`), 1-D exempt ("except 1d tensors", `gguf-py/gguf/constants.py:4649`).
  Note this path runs the **lossy** encoder (§1.4) — it is the fallback if
  passthrough proves impossible, not an alternative implementation of it.

**K3 implication:** the KDA-side special tensors already have enforced homes (convs →
F32 by list; `A_log`/`dt_bias` → F32 as 1-D; router → F32; norms → F32). Net-new K3
tensors (SiTU parameters, AttnRes pseudo-queries/norms, the MLA gate projection —
recon 04/05) will fall through to the *generic* rules: 1-D → F32, 2-D → session
ftype. Whether that default is correct for each is checkpoint-dependent (OQ8).

---

## 4. Lossless HF→GGUF checklist — what the repacker must satisfy

Each item is a hard gate; the first three decide passthrough-vs-lossy (runbook D6).

1. **Group geometry.** Group size exactly 32, groups running along the tensor's
   **contraction axis**, stored innermost (contraction-dim-last in HF, → ne[0] in
   ggml, no transpose). `ne[0] % 32 != 0` is rejected at load
   (`ggml/src/gguf.cpp:712-718`). A different group size, a global/second-level
   scale, or groups along the output axis ⇒ **no lossless repack exists** → D6 lossy
   branch, documented loudly.
2. **Nibble integrity.** Element codes move as raw 4-bit values — never through
   float. The permutation is fixed: adjacent-pair source (`v_2k` low / `v_2k+1` high
   in byte `k`) → split-half target (`qs[j] = v_j | v_{j+16} << 4`). The lossy
   encoder (`ggml-quants.c:350-382`, `quants.py:669-690`) must be unreachable from
   the MXFP4 path (dequant bypass wired as in `gpt_oss.py:17-21`).
3. **Scale integrity.** Scale bytes copied verbatim into `block_mxfp4.e`; correctness
   rests on the doubled-values/halved-scale identity (§1.3). Policy for `e = 0xFF`
   (E8M0 NaN): ggml won't decode it meaningfully (`ggml-impl.h:490`) —
   **assert-and-abort on sight**, don't write it through. Implemented:
   `repack_hf_to_ggml` refuses 0xFF (`src/repack.py:72-74`), with an
   oracle test feeding one and expecting the failure
   (`test_repack_mxfp4.py::test_rejects_nan_scale`).
4. **Fused splits at row granularity only.** Any gate/up (or gate-projection) defusing
   must slice whole rows (gpt-oss's `[:, ::2]` pattern, `gpt_oss.py:73-80`); an
   element-granularity split cuts 32-blocks and forces requantization.
5. **Pairing by name, not order.** Match `blocks`/`scales` (or `.weight`/`.scale`) by
   key like DSv4 (`deepseek.py:628-641`), not by iteration order like gpt-oss
   (`gpt_oss.py:63-81`).
6. **Write path.** Stack experts to 3-D, `add_tensor(..., raw_dtype=MXFP4)`, byte
   shape divisible by 17 so `quant_shape_from_byte_shape` derives the logical shape
   (`gguf_writer.py:358-360`, `quants.py:21-26`).
7. **Non-MXFP4 tensors routed deliberately** through §3.1's rules; any tensor
   unexpectedly *not* in MXFP4 warns loudly (gpt-oss precedent, `gpt_oss.py:98,114`)
   rather than silently changing type.
8. **File-type label** set to `MOSTLY_MXFP4_MOE` only after tensors are written
   (converter ordering constraint, §3.2).
9. **Acceptance test green** (§6) on real checkpoint bytes **before** converting
   2.8T — this is runbook Step 3(b)'s go/no-go.

---

## 5. Mapping onto our repacker (`src/repack.py`)

The repacker was written against exactly the §1 ground truth (its docstring cites the
same lines: `repack.py:1-18`). Component ↔ vendor mapping:

| Repacker | Lines | Vendor ground truth |
|---|---|---|
| `QK = 32`, `BLOCK_BYTES = 17` | `repack.py:23-24` | `ggml-common.h:214-219` |
| `KVALUES` doubled-E2M1 table | `repack.py:27-28` | `ggml-common.h:1126-1128` |
| `e8m0_to_fp32` (HF-side scale) | `repack.py:33-37` | `ggml_e8m0_to_fp32`, `ggml-impl.h:439-473` |
| `e8m0_to_fp32_half` incl. `e<2` denormals | `repack.py:40-46` | `ggml-impl.h:477-495` |
| `_split_nibbles_hf` (adjacent-pair decode) | `repack.py:49-53` | source convention per `deepseek.py:617-618` |
| `repack_hf_to_ggml` (split-half assembly, scale byte 0) | `repack.py:64-85` | `deepseek.py:619-622`; decode loop `ggml-quants.c:569-586` |
| `dequant_ggml` (bit-exact C mirror) | `repack.py:88-98` | `dequantize_row_mxfp4` |
| `unpack_ggml_to_hf` (round-trip inverse) | `repack.py:101-110` | — |

**The pluggable seam** is the source-layout decode: `_split_nibbles_hf` + verbatim
scale passthrough encode today's only observed HF convention (shared by both vendor
precedents). If K3 ships a different nibble pairing or scale placement, **only that
decode function changes**; the target assembly (`repack_hf_to_ggml`), both dequant
references, and the oracle tests stay fixed — they pin the ggml side, which K3 cannot
change. The seam is a function boundary, not yet a formal spec object; if drop day
reveals a variant layout, the variant becomes a second decode function selected by
the converter, and everything downstream is unchanged.

**Equivalence to the gpt-oss transform is proven by a reproducible test**: a numpy
port of `transform_nibble_layout` + the scale concat (`gpt_oss.py:23-46,55`) produces
byte-identical output to `repack_hf_to_ggml` on seeded random `[64, 90, 16]` u8 input
— `test_gptoss_transform_equivalence` in `tests/test_repack_mxfp4.py:88-115`,
with the numpy port living inside the test (the deepseek construction at
`deepseek.py:619-622` is identical to ours by inspection). So one repacker covers
both shipping conventions.

**Shape genericity:** `repack_hf_to_ggml` accepts arbitrary leading dims
(`[..., cols/2]` + `[..., cols/32]`, `repack.py:75-80`) — per-expert 2-D (DSv4 style)
and stacked 3-D both work; gpt-oss-style 4-D `[..., n_blocks, 16]` needs a trailing
reshape to `[..., n_blocks·16]` first.

**Closed since first draft:** the `e = 0xFF` NaN-scale policy of §4-3 —
`repack_hf_to_ggml` now refuses NaN scale bytes (`repack.py:72-74`), with an oracle
test feeding one and expecting the failure (`test_rejects_nan_scale`).

**Known gaps vs the §4 checklist** (deliberate until the checkpoint answers):

- no axis-orientation check — the caller asserts contraction-dim-last (§4-1);
- no fused-split helper (§4-4) and no converter wiring (`conversion/kimi_k3.py` does
  not exist; recon 01 §5 items #7/#8);
- pairing/naming logic is the converter's job (§4-5), out of repacker scope.

Oracle coverage (`tests/test_repack_mxfp4.py`): bit-exact dequant equality via
u32 views on random tensors (`:29-36`), scale edges `{0,1,2,126,127,128,254}`
exercising the denormal branch (`:39-44`), byte round-trip (`:47-50`), a hand-built
known vector pinning the split-half convention and the E2M1 value ladder (`:53-68`),
NaN-scale rejection (`:80-86`), and gpt-oss transform equivalence (`:88-115`). Run it
from a fresh clone: `python -m pytest tests/test_repack_mxfp4.py`.

---

## 6. Bit-exactness test definition (acceptance gate for D6 / recon 01 #8)

**Definition.** For a source pair `(blocks, scales)` and repacked bytes
`raw = repack(blocks, scales)`, the repack is *bit-exact* iff **all four** hold:

- **T1 — byte round-trip (lossless-ness):** `unpack(raw) == (blocks, scales)`,
  `np.array_equal` on uint8 — the permutation is a bijection; no nibble or scale byte
  altered.
- **T2 — dequant bit-equality (semantic preservation):**
  `dequant_hf(blocks, scales).view(uint32) == dequant_ggml(raw).view(uint32)` —
  float32 **bit patterns**, never `allclose`. Must pass on: scale edges `e ∈ {0, 1}`
  (denormal branch), `e = 254` (max valid), `e ≥ 253` with max-magnitude codes
  (`±inf` on both sides — a shared, defined outcome), and inputs containing code 8
  (−0).
- **T3 — C-convention equality on written bytes:** the GGUF tensor's stored bytes,
  decoded with the C-mirroring reference (`MXFP4.dequantize_blocks`,
  `gguf-py/gguf/quants.py:692-706`, which mirrors `dequantize_row_mxfp4`), u32-equal
  to `dequant_hf` of the source — catches writer-side reshapes/stacking, not just the
  in-memory repack.
- **T4 — on-checkpoint spot check (drop day, runbook Step 3(b)):** T1–T3 on one real
  pulled expert tensor before any full-scale conversion; the verdict goes to
  FINDINGS.md and gates D6.

**One documented asymmetry:** code 8 (−0) dequants to **+0.0** under both our HF
reference (`repack.py:30`: `0 × 0.5`) and ggml (int8 table `0`) — internally
consistent, so T2/T3 are unaffected — but an *external* OCP-faithful decoder (e.g. a
torch/triton reference) may emit **−0.0** there. Any comparison against third-party
dequants must therefore compare through our reference pair, or treat `±0` as equal
while still requiring T1's byte equality (which preserves the −0 *code* exactly).

Failure of any T-item ⇒ the layout deviates from both shipping precedents ⇒ D6's
translation branch: lossless variant-decode if the deviation is still a permutation;
dequant–requant only if provably nothing else exists, with the lossy step documented
loudly (runbook §Step 4, D6).

---

## Open questions — what only the K3 checkpoint can settle

1. **Serialization form.** Does K3's safetensors carry MXFP4 as paired u8
   `blocks`/`scales` tensors at all (runbook 06 OQ4), and under which naming —
   gpt-oss-style `*_blocks`/`*_scales`, DSv4-style `.weight`/`.scale`, or new? What
   `quantization_config.quant_method` string gates detection (`base.py:827` keys on
   the literal `"mxfp4"`)?
2. **MXFP4 coverage.** Which tensors are MXFP4 — routed experts only (both
   precedents), or also shared experts / dense FFN / attention projections / the
   vision tower? vLLM's post says only "MXFP4 weights in the provided release
   configuration" (recon 05 A11). Anything beyond stacked MoE breaks the
   `ne[2] > 1` assumptions baked into the tooling (§3.2).
3. **Group geometry** (§4-1): 32-element groups along the contraction axis,
   contraction-dim-last? This is the single passthrough-vs-lossy fork.
4. **Nibble pairing** (§4-2): adjacent-pair like both precedents, or a variant —
   decides whether `_split_nibbles_hf` survives unchanged or gains a sibling.
5. **Scale semantics** (§4-3): raw E8M0, bias 127, no second-level/per-tensor global
   scale? (NVFP4 checkpoints needed a separate runtime-multiplied scale2 tensor —
   `conversion/base.py:656+` — an MXFP4 analogue would add converter+graph work.) Do
   real scale bytes include `0xFF`?
6. **Fusion layout.** Is gate/up fused row-interleaved (gpt-oss) or stored separately
   (DSv4)? Is the new MLA gate projection (recon 05 A6) fused with anything, and is
   it MXFP4 or bf16?
7. **Expert storage shape.** Stacked `[E, …]` tensors or per-expert entries — picks
   the converter skeleton to clone (gpt-oss vs DSv4) and whether repack input is 4-D
   or 2-D.
8. **Precision of K3-specific small tensors.** SiTU parameters, AttnRes
   pseudo-queries/norms, `e_score_correction`-style router biases: their source
   dtypes and whether the generic rules (§3.1) place them correctly, or K3 needs
   additions to the always-F32 list the way KDA convs did (`base.py:919-922`).
9. **Divisibility.** Is every MXFP4 tensor's contraction dim ≡ 0 (mod 32) (gpt-oss's
   2880 was; K3's expert dims are unpublished — recon 05 A15/A16)? A violation is
   rejected at load (`gguf.cpp:712-718`) and would force padding decisions with no
   precedent.
10. **Embedding/output budget.** Converter default leaves `token_embd`/`output` at
    bf16 (§3.1) while the requantize tool's MXFP4_MOE preset uses Q8_0
    (`llama-quant.cpp:450-452`) — at 2.8T-scale vocab dims, which is wanted is a
    size/quality call to make once shapes are known.
