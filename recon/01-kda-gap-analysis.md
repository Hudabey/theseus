# 01 — KDA vs gated-deltanet gap analysis

Scope: compare **Kimi Delta Attention (KDA)** — the attention in the July-27 Kimi K3 target —
against the **gated-deltanet** paths already in the llama.cpp family. Reference sources:
`vendor/kimi-linear` (paper), `vendor/fla` (KDA kernels), `vendor/llama.cpp` and
`vendor/ik_llama.cpp` (C/GGML implementations). Every claim carries a file:line ref.

> **Anchor.** KDA is a gated-deltanet variant. It is **not** the MLA attention used in Kimi
> K2 / K2.5. The Kimi-K2 code in these repos (tokenizer, chat template, tool parser in
> `vendor/ik_llama.cpp`) is unrelated and is **not** a reference point here. The correct
> comparison is KDA vs the Qwen3-Next gated-deltanet path — and, as it turns out, vs the
> **Kimi-Linear** KDA path that mainline llama.cpp already ships (see §3).

---

## TL;DR — the headline finding (revises the initial working hypothesis)

**The initial working hypothesis is already stale for mainline llama.cpp.** The assumed
strategy — "adapt existing gated-deltanet support (built for Qwen3-Next) to KDA" — describes
work that is **already done** in `vendor/llama.cpp`:

- The core op `ggml_gated_delta_net` already has a **first-class per-channel (KDA) gate mode**,
  documented in the header and supported by **every backend** (CPU, CUDA, Metal, Vulkan, SYCL,
  OpenCL, WebGPU, Hexagon, OpenVINO, ggml-et) — [`vendor/llama.cpp/ggml/include/ggml.h:2569`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/ggml/include/ggml.h#L2569),
  `vendor/llama.cpp/ggml/src/ggml.c:6284-6286`.
- A complete `LLM_ARCH_KIMI_LINEAR` graph runs KDA end-to-end (per-channel gate, L2-norm,
  `-exp(A_log)` decay, sigmoid β, sigmoid output gate) — [`vendor/llama.cpp/src/models/kimi-linear.cpp:288-373`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/src/models/kimi-linear.cpp#L288-L373).
- A converter exists — [`vendor/llama.cpp/conversion/kimi_linear.py:15,175-176`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/conversion/kimi_linear.py#L175-L176).

So for mainline, KDA itself is **not** the gap. The real K3 gaps are: **(a) Attention
Residuals** (absent from both repos — a genuinely new mechanism), **(b) a new K3 arch +
hybrid-layout signaling**, **(c) native MXFP4 → GGUF without re-quant**, and **(d)** confirming
K3's KDA numerics match Kimi-Linear's. `vendor/ik_llama.cpp` is a different story — it is far
behind (scalar gate only, no KDA, no converter; see §3/§4).

---

## 1. The recurrence, side by side

Notation: `k,q` key/query (dim `d_k`), `v` value (dim `d_v`), `β` write strength, `S` the
matrix state `[d_k × d_v]`, `α` the forget gate. All three implementations apply **decay
first, then the delta write against the already-decayed state**.

### Gated DeltaNet (GDN) — Qwen3-Next

Paper form (`vendor/kimi-linear/tech_report.pdf` p.3, "Gated DeltaNet as Weight Decay"):

    S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ ,   α_t ∈ [0,1]  (scalar per head)

fla naive reference (`vendor/fla/fla/ops/gated_delta_rule/naive.py:50-59`), verbatim structure:

```python
h = h.clone() * g[:, :, i].exp()[..., None, None]      # scalar decay, whole state
b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)        # correction vs decayed state
b_v = b_v * b_beta[..., None]
h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)   # rank-1 write
o[:, :, i] = einsum('bhd,bhdm->bhm', b_q, h)
```

Gate shape `g:[B,T,H]` — **one scalar per head** (`.../naive.py:27-33`).

### Kimi Delta Attention (KDA) — K3 / Kimi-Linear

Paper Eq. 1 (`vendor/kimi-linear/tech_report.pdf` p.4):

    S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ ;   o_t = S_tᵀ q_t

`Diag(α_t)` is a **per-channel** diagonal gate (one decay value per key-channel), vs GDN's
scalar `α_t`. fla naive reference ([`vendor/fla/fla/ops/kda/naive.py:59-63`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/ops/kda/naive.py#L59-L63)), verbatim:

```python
S = S * g_i[..., None].exp()                                              # per-channel decay
S = S + torch.einsum('b h k, b h v -> b h k v',
                     b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
```

Gate shape `g:[B,T,HV,K]` — **a vector over the key dimension per (value-)head**
(`.../kda/naive.py:24-33`). Verified first-hand in the decode kernel
(`vendor/fla/fla/ops/kda/fused_recurrent.py:174-197`): decay `b_h *= exp(b_gk)` broadcast per
K-channel, then `v − kᵀS`, β-sigmoid, rank-1 write — exactly Eq. 1.

### The gate itself (both)

Both compute the decay from a projection through softplus, scaled by a per-head learned `A`:

    g = −exp(A_log) · softplus( f_proj(x) + dt_bias )      (so g ≤ 0, exp(g) ∈ (0,1])

- GDN: `A_log`, `dt_bias` are **per-head scalars** (`vendor/fla/fla/ops/gated_delta_rule/gate.py:43-45`).
- KDA: `A_log` is per-head `[H]`, but **`dt_bias` is per-head-per-channel `[H*K]`**, and
  `f_proj` is a low-rank two-matrix bottleneck (see §2) — `vendor/fla/fla/ops/kda/gate.py:49-54`,
  `vendor/fla/fla/layers/kda.py:166-185`.

### The one-line diff

The recurrence math is **identical** except the forget gate goes from **scalar-per-head**
(`exp(g)` scales the whole state) to **vector-over-key-channels** (`Diag(exp(g))` scales each
state row). Everything downstream (per-channel `dt_bias`, GVA head expansion, chunkwise output
kernel choice) follows from that single change.

---

## 2. What KDA changes vs GDN

### (a) Channel-wise vs head-wise gating — the core change

- GDN: `g:[B,T,H]`, decay = `exp(g)` applied to the entire `S` (`gated_delta_rule/naive.py:54`).
- KDA: `g:[B,T,HV,K]`, decay = `Diag(exp(g))`, i.e. `S[i,:] *= exp(g[i])` per key-channel `i`
  (`kda/naive.py:61`; C-side `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:10842-10850`).
- Consequences: `dt_bias` becomes per-channel `[H*K]` (`fla/layers/kda.py:180-185`); the gate
  projection `f_proj` is **low-rank**: `hidden → head_v_dim → num_v_heads*head_k_dim`, two
  bias-free matrices (`fla/layers/kda.py:166-171`). GVA (grouped value attention): q/k at `H`,
  v/g/β at `HV` with `q,k` repeat-interleaved by `G=HV//H` (`fla/ops/kda/naive.py:52-53`;
  default `num_v_heads=None ⇒ =num_heads`, i.e. no GVA — `fla/models/kda/configuration_kda.py:26`).

### (b) DPLR structure

Tech report §3.2 (`tech_report.pdf` p.5): KDA aligns with the generalized **DPLR**
(Diagonal-Plus-Low-Rank) form `S_t = (D − a_t b_tᵀ) S_{t−1} + k_t v_tᵀ`, but KDA **ties both
low-rank vectors to k** (`a = b = k`, with `D = Diag(α)`). Binding `a,b` to `k` "reduces the
number of second-level chunk matrix computations from four to two" and removes three matmuls,
giving ~2× operator efficiency over general DPLR (`tech_report.pdf` p.5, Fig. 2). Practical
takeaway for us: KDA is **not** a general DPLR kernel — it is the rank-1-tied special case, so
the existing delta-net rank-1 write is the right primitive; no general `a≠b` path is needed.

### (c) Chunkwise computation

KDA keeps the WY-representation / UT-transform chunkwise scheme (paper Eqs. 2–9,
`tech_report.pdf` p.4) but must carry the **per-channel** cumulative gate:

- **fla**: `chunk_kda` uses a **vector** gate cumsum in log2 space
  (`vendor/fla/fla/ops/kda/chunk_fwd.py:45-56`, `kda/gate.py` vector cumsum kernel), and — because
  the gate is per-channel — routes the output through **GLA's** vector-gate output kernel
  `chunk_gla_fwd_o_gk` (`kda/chunk_fwd.py:115-126`) instead of GDN's scalar `chunk_fwd_o`. The
  inter-chunk state recurrence kernel `chunk_gated_delta_rule_fwd_h` is **shared** with GDN, but
  KDA passes the vector gate `gk=g` and gate-scaled `kg` (`kda/chunk_fwd.py:94-106`).
- **mainline llama.cpp**: `build_delta_net_chunking` branches on a `kda` bool; chunk size is
  **`CS = kda ? 16 : 64`** (`vendor/llama.cpp/src/models/delta-net-base.cpp:61`) and the decay
  mask is built per-channel via `ggml_sub(g_cs_j, g_cs_i) → ggml_tri → ggml_exp`
  (`.../delta-net-base.cpp:94-124`), using `ggml_cumsum` (`:89`) and `ggml_solve_tri` for the
  `(I+tril)⁻¹` inverse (`:160-167`). The scalar (GDN) path is the `else` branch (`:125-145`).

---

## 3. Implementation-status matrix (the reframe)

| Capability | `vendor/llama.cpp` (mainline) | `vendor/ik_llama.cpp` |
|---|---|---|
| Delta-net op | `ggml_gated_delta_net` — `ggml/include/ggml.h:2576` | `ggml_delta_net` — `ggml/include/ggml.h:2589` |
| **Per-channel (KDA) gate** | **Yes** — `g.ne[0]==S_v` toggles KDA; `ggml.c:6284-6286` | **No** — gate asserted `[n_tokens,1,H_v,n_seqs]`, scalar only; `ggml.c:10131` |
| Backends w/ delta-net | CPU, CUDA, Metal, Vulkan, SYCL, OpenCL, WebGPU, Hexagon, OpenVINO, ggml-et (all handle KDA) | **CPU + CUDA only** (`ggml-cuda/delta-net.cu`); no Metal/Vulkan/SYCL |
| KDA gate math (`-exp(A_log)·softplus(·)`) | In-graph — `kimi-linear.cpp:302-314` | In-graph but scalar `A` — `llama-delta-net.cpp:276-280` |
| Kimi-Linear arch (KDA+MLA) | **Yes, complete** — `src/models/kimi-linear.cpp`, `LLM_ARCH_KIMI_LINEAR` `llama-arch.cpp:142` | **Not found** |
| Qwen3-Next arch (GDN) | Yes — `src/models/qwen3next.cpp` | Yes — `src/graphs/build_qwen3next.cpp` |
| Converter (HF→GGUF) | `conversion/kimi_linear.py` + `conversion/qwen.py:271` | **None** — consumes GGUF only |
| MXFP4 as a ggml type | Yes — `GGML_TYPE_MXFP4=39` `ggml/include/ggml.h:429`; ftype `MOSTLY_MXFP4_MOE=38` `gguf-py/.../constants.py:4649` | Yes, first-class + optimized GEMM — `ggml/include/ggml.h:427`, `iqk/iqk_gemm_legacy_quants.cpp:641` |
| **Attention Residuals** | **Not found** (only standard pre-norm residuals) | **Not found** |
| K3 / KimiK3 references | **None anywhere** | **None anywhere** |

Verified first-hand: `ggml.h:2569` (KDA gate doc), `kimi-linear.cpp:302-340` (KDA path),
`conversion/kimi_linear.py:175-176` (`A_log → -exp(A_log)`, `dt_bias → dt_proj.bias`), absence
of attnres/K3 (grep clean), MXFP4 present. The ik-side scalar gate and clamp verified at
`ggml.c:10131`, `ggml.c:23494/23525`, and `llama-delta-net.cpp:276-280`.

---

## 4. Reusable as-is / needs modification / missing

Framed for the **K3 target**, primarily against mainline (the viable base).

### Reusable as-is (mainline)

- **The delta-net op and all its backend kernels.** `ggml_gated_delta_net` already does the KDA
  per-channel recurrence on every backend (`ggml.c:6255-6309`; CPU `ops.cpp:10831-10872`; matrix
  in §3). If K3's KDA is numerically identical to Kimi-Linear's, these **may be reusable with
  little or no recurrence-kernel work — provided K3 matches the supported shapes and conventions**.
- **The KDA graph primitives** `build_delta_net` / `_chunking` / `_autoregressive` / `_fused`
  (`src/models/delta-net-base.cpp:16-447`) — chunk size 16, per-channel decay mask, L2-norm on
  q/k, `-exp(A_log)` decay, sigmoid β/output-gate.
- **The hybrid recurrent+attention memory** (`llama_memory_hybrid`, default filters
  `filter_recr = is_recr(il)` — `src/llama-memory-hybrid.cpp:48-49,62-63`).
- **MXFP4 quant type + block layout** (`GGML_TYPE_MXFP4`) — the target format already exists as a
  loadable/runnable type on both repos.

### Needs modification

- **New arch `LLM_ARCH_KIMI_K3`** paralleling `LLM_ARCH_KIMI_LINEAR`
  (`llama-arch.{h,cpp}`, hparams read in a new `src/models/kimi-k3.cpp` cloned from
  `kimi-linear.cpp:4-167`). Low risk if KDA is unchanged; the graph is a copy with the
  residual + layout changes below.
- **Hybrid-layout signaling.** Kimi-Linear marks KDA layers via `n_head_kv(il)==0`
  (`kimi-linear.cpp:17-18`); Qwen3-Next uses `full_attention_interval` / explicit
  `attention.recurrent_layers` (`qwen3next.cpp:17-22`). K3's 2.8T layout (paper's reference
  Kimi-Linear is 3:1 KDA:MLA — `tech_report.pdf` p.6) must be mapped to one of these; may need a
  new KV key.
- **Converter** `conversion/kimi_k3.py` cloned from `conversion/kimi_linear.py` — same
  `A_log → -exp(A_log)` (`:175-176`), `dt_bias → dt_proj.bias` (`:177-179`), conv1d reshape
  (`:158-170`), plus the MXFP4 passthrough (below) and any new attnres tensors.
- **Full-attention layer type.** Kimi-Linear's global layers are **MLA with NoPE**
  (`kimi-linear.cpp:242,374-473`; `tech_report.pdf` p.6). K3's are externally stated to be
  **MLA every four layers** (vLLM preview — recon 05 A5 [V]), so the Kimi-Linear MLA path is
  the reuse target; the preview also states a **new gate projection** on K3's MLA (recon 05
  A6) that is net-new graph work. Remaining unknowns: placement offset, tensor layout, and
  the gate wiring — see Open questions.

### Missing entirely (must be built)

- **Attention Residuals.** Nothing in either repo. In fla this is [`fla/ops/attnres/`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/ops/attnres/naive.py#L56-L75) (a depth-axis
  softmax over RMS-normed residual sources: `naive.py:56-75`) wired into the KDA model block
  ([`fla/models/kda/modeling_kda.py:88-167`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/models/kda/modeling_kda.py#L88-L167)), **not** a standard residual add. This is the
  largest net-new piece, but the correctness-first path is **graph composition from existing
  ggml ops** (`rms_norm`, `mul_mat`, `concat`, `soft_max`, `view`, `mul`, `add` — recon 04 §5,
  "no new kernel is required for a day-one correct path"); a **fused op** (modeled on
  `fla/ops/attnres/fused.py:38-121`) is conditional on the released layout (full-mode's
  O(layers²) node growth — recon 04 §5) and on measured performance need. Net-new either way:
  the tensors (`attn_res_proj`, `attn_res_norm`, …), the prefix-sum stream discipline, and
  converter support. **Note:** the mechanism fla implements is a Kimi-Linear-era construct;
  whether K3 uses exactly this is an open question, but *some* attention-residual mechanism is
  externally stated for K3 ("AttnRes across depth" — vLLM preview, recon 05 §2) and exists in
  neither llama.cpp repo.
- **Native MXFP4 → GGUF without re-quant.** The type exists, but no Kimi/Qwen3-Next conversion
  path preserves native MXFP4 — mainline even forces KDA conv weights to F32
  (`conversion/base.py` comment "Kimi KDA conv weights should be F32"). A passthrough path that
  keeps already-MXFP4 MoE tensors in `GGML_TYPE_MXFP4` (no dequant/requant) must be written.
- **On `vendor/ik_llama.cpp` specifically:** KDA per-channel gate is entirely absent — the scalar
  `decay = expf(min(g,50))` multiplies the whole state (`ggml.c:23494,23525`), gate asserted
  scalar (`ggml.c:10131`). Bringing KDA to ik would mean rewriting the gate shape + decay in
  **three** kernels (`ggml.c`, `iqk/iqk_mul_mat.cpp:1657-1710`, `ggml-cuda/delta-net.cu:155-160`)
  plus the `build_beta_gate` projection — i.e. re-doing what mainline already did. **Recommend
  mainline as the base**; treat ik only as a source of MXFP4 GEMM kernels if needed.

---

## 5. Concrete change list (K3 on mainline llama.cpp)

Difficulty: **S** = small/mechanical, **M** = moderate, **L** = large/novel.
Ordered roughly by dependency.

| # | Change | Files (mainline) | Difficulty | Notes |
|---|---|---|---|---|
| 1 | Confirm K3 KDA == Kimi-Linear KDA (chunk 16, sigmoid gate/β, `-exp(A_log)`) | — (config diff on release) | **S** | Gates whether §5.2–5.4 kernel work is zero. Blocking. |
| 2 | Register `LLM_ARCH_KIMI_K3` + KV keys | `llama-arch.{h,cpp}` | **S** | Clone Kimi-Linear entries. |
| 3 | New graph `src/models/kimi-k3.cpp` | clone of `kimi-linear.cpp:4-528` | **M** | Reuses `build_delta_net`; edits = layout + residuals. |
| 4 | Hybrid-layout signaling for K3 | `kimi-k3.cpp` load hparams, `llama-hparams.cpp` | **M** | Map 2.8T layout to `is_recr_impl[]`; maybe new KV key. |
| 5 | **Attention Residuals** graph + tensors | graph-composed from existing ops in `kimi-k3.cpp` (recon 04 §5); `llama-arch.cpp` tensor enums; fused op only if full-mode layout or perf forces it | **M–L** | Net-new. Correctness-first: composition per recon 04 §5; fused-op contingency per recon 04 §5 / runbook D3. |
| 6 | Full-attn layers (MLA, externally stated — recon 05 A5) + new gate projection | `kimi-k3.cpp` | **S–M** | Reuse the Kimi-Linear MLA path; gate projection (recon 05 A6) is net-new; offset/layout pending (OQ3). |
| 7 | Converter `conversion/kimi_k3.py` | clone `conversion/kimi_linear.py` | **M** | `A_log`/`dt_bias`/conv reshape reusable; add attnres tensors. |
| 8 | **Native MXFP4 passthrough** in converter | `conversion/kimi_k3.py`, `conversion/base.py`, `gguf-py` | **L** | Keep MXFP4 MoE tensors un-dequantized; verify block layout `QK_MXFP4=32`. |
| 9 | gguf-py constants/tensor-mapping for K3 | `gguf-py/gguf/constants.py`, `tensor_mapping.py` | **S** | Clone Kimi-Linear block `constants.py:4392-4432`, `tensor_mapping.py:893-912`. |
| 10 | End-to-end validation vs fla reference | new test | **M** | Compare against `fla` naive/`chunk_kda` on real K3 weights. |

Net: if #1 confirms KDA is unchanged, the delta-net **kernel work may be near zero**; the real cost
concentrates in **#5 (Attention Residuals, M–L)** and **#8 (MXFP4 passthrough, L)**, with the rest
being arch-registration and converter cloning.

---

## Open questions

1. **Does K3's KDA differ numerically from Kimi-Linear's KDA?** Chunk size 16, sigmoid output
   gate, sigmoid β, `-exp(A_log)` per-channel decay, `head_dim=128` — are these identical for
   K3, letting `ggml_gated_delta_net` and `build_delta_net` be reused verbatim? (Gates §5.) Not
   determinable until the weights/config release (announced for July 27).
2. **What exactly are K3's "Attention Residuals," and where do they attach?** fla's `attnres`
   (depth-axis softmax over RMS-normed residual sources, `fla/ops/attnres/naive.py:56-75`, wired
   in `fla/models/kda/modeling_kda.py:88-167`) is the closest reference, but the
   announcement-level phrasing ("AttnRes across depth" — vLLM preview, recon 05 §2) is not
   code-confirmed for K3. Is it fla-style attnres, a per-layer skip into recurrent state, or
   something new? Nothing analogous exists in either repo.
3. ~~MLA or GQA for K3's periodic full-attention layers?~~ **Resolved externally: MLA, every
   four layers** (vLLM preview — recon 05 A5 [V]); the Kimi-Linear **MLA-with-NoPE** path
   (`kimi-linear.cpp:242,374-473`) is the reuse target for #6. Still open: the **placement
   offset** (93 mod 4 ≠ 0 — recon 05 A12), the **tensor layout**, and the **new MLA gate
   projection's wiring** (recon 05 A6 / 05-OQ5).
4. **How is K3's hybrid layout signaled** — a Kimi-Linear-style `n_head_kv==0` per-layer mark, a
   Qwen3-Next `full_attention_interval` / explicit `attention.recurrent_layers` list, or a new
   key? (The every-four-layers-MLA statement — recon 05 A5 — matches Kimi-Linear's published
   3:1 ratio, `tech_report.pdf` p.6; the in-checkpoint signaling mechanism and the placement
   offset remain unknown.)
5. **Does native MXFP4 survive as MoE weights unchanged?** Is K3's MXFP4 block layout compatible
   with ggml `block_mxfp4` (`QK_MXFP4=32`, `e:uint8 + qs[16]`)? If yes, #8 is a passthrough; if
   the scale/packing differs, a translation step is needed. Mainline currently forces KDA conv
   weights to F32 — need to confirm only conv (not MoE) is affected.
6. **Not verified against the Triton chunk kernel directly.** §2(c) chunkwise details for fla are
   from the naive reference + `chunk_fwd.py` call graph, not from the 36 KB
   `fla/ops/kda/chunk_intra.py` kernel. If exact intra-chunk `Aqk`/`Akk` formulas matter, read
   that kernel.
7. **Cross-repo tensor-name consistency** if ik_llama.cpp is ever targeted: ik expects short
   names (`blk.%d.ssm_a`, `ssm_ba`) and has no converter; mainline emits Kimi-specific names
   (`ssm_f_a`, `ssm_g_b`, …). A GGUF produced for mainline K3 will not load in ik without a name
   map. (Out of scope unless ik becomes a target.)
