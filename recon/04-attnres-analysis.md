# 04 — Attention Residuals (AttnRes) ground truth from vendor/fla

Scope: extract the exact AttnRes mechanism from `vendor/fla` — the op
(`fla/ops/attnres/`) and its wiring into the KDA model
(`fla/models/kda/modeling_kda.py`) — as the reference for the net-new C/GGML work
identified in `recon/01-kda-gap-analysis.md` §4/§5 (item #5, difficulty L). Every claim
carries a file:line ref. The paper the op cites is arXiv 2603.15031
(`vendor/fla/fla/ops/attnres/naive.py:31`).

Reference surface: `naive.py` (semantic reference), `fused.py` (Triton kernel — the
version the model actually calls), `backends/gluon.py` (opt-in Gluon port, numerically
identical, `backends/gluon.py:8-17`), `fla/tests/ops/test_attnres.py` (frozen parity test),
`fla/models/kda/{modeling,configuration}_kda.py` (wiring). The identical wiring also
exists in `fla/models/abc/modeling_abc.py:88-167` — the mechanism is model-agnostic;
nothing about it is KDA-specific. `vendor/kimi-linear` contains **zero** attnres
references (grep clean), so fla is the only code-level ground truth until K3 drops.

> **Scope.** Every claim in this doc describes **fla's reference implementation** —
> the wiring, the tensors, the caching behavior, the block arithmetic. Whether K3
> follows any of it is checkpoint-dependent; the Open questions section is the list
> of what transfers only after verification.

---

## TL;DR

In fla's wiring, AttnRes replaces the **read** of the residual stream, not the write. At each sub-layer,
instead of `norm(prefix_sum)` as the sub-layer input, the model keeps a **list of
residual sources** (embedding + per-block partial sums + the running intra-block sum)
and forms the sub-layer input as a **per-token softmax mixture over that list along the
depth axis**:

    k_i = RMSNorm(v_i)                      (shared rms_weight, per sub-layer)
    p   = softmax_i( q · k_i · scale )      (q = learned per-sub-layer pseudo-query, one vector)
    o   = Σ_i p_i · v_i                     (mixed over the *raw*, un-normalized sources)
    input = RMSNorm_out(o)                  (the block's ordinary prenorm, folded in)

The sub-layer's output is then **added** into a running prefix sum exactly as in a
vanilla pre-norm transformer — the stored pieces sum algebraically to the full
stream, but the AttnRes *read* of them is a learned convex mixture, not a lossless
reconstruction; only the write path stays a plain sum. One extra top-level mixture
replaces the final `norm(h)` before the LM head.
Everything decomposes into existing ggml ops (`rms_norm`, `mul`, `mul_mat`, `concat`,
`soft_max`, `view`, `add`) — see §5.

---

## 1. Exact forward math (op level)

Semantic reference: [`fla/ops/attnres/naive.py:56-80`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/ops/attnres/naive.py#L56-L80). Inputs
(`naive.py:33-48` / `fused.py:522-542`):

- `query` — one vector `[D]` (accepted as `[D]` or `[D,1]`, flattened at
  `naive.py:68`). This is the **pseudo-query**: a per-sub-layer *parameter*, not a
  projection of the hidden state.
- `residuals` — a Python sequence of `L ≥ 1` same-shape tensors `[..., D]`
  (`naive.py:56-57`). L is the **depth axis** — number of residual sources, not tokens.
- `rms_weight [D]` — RMSNorm scale used to build keys.
- `output_rms_weight [D]` (optional) — if set, one more RMSNorm is applied to the mixed
  output before returning; this **folds the block's ordinary prenorm** (`attn_norm` /
  `mlp_norm`) into the op (`naive.py:40-42`, applied at `naive.py:72-73`).
- `rms_eps` (shared by both norms, `naive.py:43-44`), `scale` on logits before softmax
  (default **1.0**; `naive.py:45-46`).

Step by step (`naive.py:63-75`, all math in fp32 with a single final downcast,
`naive.py:65-66,75`):

1. **Stack** the L sources: `stacked [L, N, D]` where `N` = product of leading dims
   (`naive.py:63`). The stack is an artifact of the reference impl; the fused kernel
   takes the L tensors as separate pointers (`fused.py:301-308`) — no concat ever
   happens in the model path.
2. **Keys** = RMSNorm of each source: `k = rms_norm(v) * rms_weight` (`naive.py:67`).
   The *values* stay un-normalized.
3. **Scores/softmax**: `p = softmax_L( (k · (q*scale)) )` — a dot product per
   (source, position) giving logits `[L, N]`, softmax **over the depth axis `L`**
   (dim 0), per token independently (`naive.py:68`). There is no interaction across
   positions — this is not sequence attention.
4. **Mix**: `o = Σ_l p_l · v_l` `[N, D]` (`naive.py:69`), i.e. a convex combination of
   the **raw** sources weighted by the depth softmax.
5. Optional **output RMSNorm** with `output_rms_weight` (`naive.py:72-73`), then
   downcast to input dtype (`naive.py:75`). `return_weights` optionally exposes `p`
   (`naive.py:77-79`); the model never uses it.

The fused Triton kernel ([`fused.py:38-121`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/ops/attnres/fused.py#L38-L121)) computes the same thing with an online
softmax over L-tiles (`fused.py:68-100`), one program per position `i_n`. Two identities
in it worth keeping for the C port:

- The logit is computed as `(q ⊙ rms_weight) · v · rstd` with
  `rstd = rsqrt(mean(v²) + eps)` (`fused.py:65,90-91`) — i.e. RMSNorm can be applied as
  a *scalar* after the dot product, and `q ⊙ rms_weight` is a **weights-only product**
  that can be precomputed at load time.
- The output norm is folded post-mix (`fused.py:116-119`); `o_pre` (the pre-norm mix)
  is only kept for backward (`fused.py:112-114`) — irrelevant for inference.

Kernel-only constraints that do **not** carry semantic meaning: sources must be
16-byte aligned and are flattened contiguous (`fused.py:307`, `fused.py:557`); the
pointer tuple is padded to `L2 = max(8, next_pow2(L))` as a compile-signature trick
(`fused.py:301-308`). Tested envelope: L up to 29, T up to 8000, D up to 7186, fp16 and
fp32, with and without folded output norm (`fla/tests/ops/test_attnres.py:19-39`).

---

## 2. Wiring in the KDA model — where sources come from

[`fla/models/kda/modeling_kda.py`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/models/kda/modeling_kda.py). Enabled iff `config.attnres_block_size is not None`
(`modeling_kda.py:88`, [`configuration_kda.py:48,78`](https://github.com/fla-org/flash-linear-attention/blob/d1ce07369d581813553f30a750af3b6b5f9af6a9/fla/models/kda/configuration_kda.py#L48)).

### The stream decomposition

With attnres on, the tensor passed between layers is **not** the full residual stream;
it is the **running intra-block prefix sum** (`modeling_kda.py:117-120,176-179`), and
the rest of the stream lives in `attnres_states`, a list of completed-block partial
sums threaded through the layer loop (`modeling_kda.py:321-325,333-341`). The pieces
sum algebraically to the stream (`Σ attnres_states + prefix_sum` = what a vanilla
stream would carry); the AttnRes read, however, is a learned convex mixture of the
pieces, not that sum.

Per sub-layer (attn side shown; mlp side is symmetric):

1. On entry, `hidden_states` **is** the running `prefix_sum` (`modeling_kda.py:118-120`).
2. Build `residuals = [*attnres_states, prefix_sum]` (`modeling_kda.py:130`).
3. If this sub-layer is a **block boundary**, the just-completed block's sum is
   promoted into `attnres_states` and `prefix_sum` resets to `None`
   (`modeling_kda.py:131-134`; mlp side `modeling_kda.py:158-160`).
4. The sub-layer input = `fused_attnres(query=attn_res_proj.weight, residuals,
   rms_weight=attn_res_norm.weight, output_rms_weight=attn_norm.weight)`
   (`modeling_kda.py:135-141`) — note the block's ordinary prenorm weight rides in as
   the folded output norm, and **no `scale` is passed → scale = 1.0**.
5. The sub-layer output is **added** into `prefix_sum` (`modeling_kda.py:156` for attn
   output, `modeling_kda.py:176-179` for mlp output). Plain add — no mixing on the
   write path.

First sub-layer special case (`modeling_kda.py:121-128`): `attnres_states is None` →
sources would be `[embeddings]` alone, and an L=1 softmax is identically 1, so the code
short-circuits to `attn_norm(prefix_sum)` and seeds `attnres_states = [embeddings]`.
**The token embedding is always source 0.**

### Top-level aggregation

After the layer loop, one final mixture with model-level parameters replaces the final
norm's input: `residuals = [*attnres_states, hidden_states]`,
`fused_attnres(query=res_proj.weight, rms_weight=res_norm.weight,
output_rms_weight=self.norm.weight)` (`modeling_kda.py:346-356`). The final `RMSNorm`
weight is folded in the same way, so the LM head consumes a normed depth mixture, not
`norm(prefix_sum_total)`.

### Worked trace, `attnres_block_size = 4` (block = 2 transformer layers = 4 sub-layers)

| point | `attnres_states` | `prefix_sum` at read | read sources |
|---|---|---|---|
| L0 attn (global sub-layer 0, boundary) | `[emb]` (seeded) | — | bypass: `attn_norm(emb)` |
| L0 mlp (idx 1) | `[emb]` | `attn0` | `[emb, attn0]` |
| L1 attn (idx 2) | `[emb]` | `attn0+mlp0` | `[emb, attn0+mlp0]` |
| L1 mlp (idx 3) | `[emb]` | `attn0+mlp0+attn1` | `[emb, …]` |
| L2 attn (idx 4, **boundary**) | `[emb, B0]` where `B0 = attn0+mlp0+attn1+mlp1` | reset | `[emb, B0]` |
| L3 mlp (idx 7) | `[emb, B0]` | `attn2+mlp2+attn3` | `[emb, B0, …]` |
| top-level | `[emb, B0, B1]` | `B2`-so-far | `[emb, B0, B1, B2]` |

Sources are **disjoint partial sums** of sub-layer outputs (plus the embedding).
Zero-initialized pseudo-queries (`modeling_kda.py:219-222`, "paper §5") produce
uniform depth weights — but a uniform softmax yields the depth **mean** of the
sources, not their sum, and the RMSNorm that follows removes that 1/L scale only
approximately (finite eps). So the init closely preserves the *direction* of the
ordinary accumulated residual while not being strictly identical to a vanilla
transformer.

---

## 3. Block vs full mode — what defines a boundary (fla's definition)

In fla's config, `attnres_block_size` ∈ {`None`, `1`, even integer ≥ 2} (`configuration_kda.py:83-88`):
`None` = off, `1` = **full mode**, even `N` = block mode with **`N/2` transformer
layers (= `N` sub-layers) per block** (`configuration_kda.py:86-87`).

Sub-layers carry a global index: attn of layer `i` is `2i`, mlp is `2i+1`. A sub-layer
**starts a new block** iff its global index ≡ 0 mod `attnres_block_size`
(`modeling_kda.py:94-102`). Two consequences verified from the arithmetic:

- **Even `N`**: `2i+1` is odd, so `attnres_is_mlp_boundary` is *never* true
  (`modeling_kda.py:102`) — snapshots happen only at attn sub-layers of layers with
  `layer_idx % (N/2) == 0`. Source count at the top level =
  `1 (emb) + ⌈num_layers/(N/2)⌉ − 1 (completed-block sums) + 1 (running sum)`.
- **`N = 1` (full mode)**: every sub-layer is a boundary, so every individual sub-layer
  output becomes its own permanent source, `prefix_sum` resets every step, and the
  tensor carried between sub-layers degenerates to just the previous sub-layer's
  output. Sub-layer `l` reads over `l+1` sources = `[emb, out_0, …, out_{l−1}]`; the
  top level reads over `2·num_layers + 1` sources.

So "Block AttnRes" = the even-`N` mode; the granularity of a source is a block partial
sum, and the boundary is purely index arithmetic — there is no learned or content-based
blocking.

---

## 4. Learned tensors (fla's inventory — K3's names/granularity are checkpoint-dependent)

In fla's model, per `KDABlock` when attnres is on (`modeling_kda.py:90-93`), plus model level
(`modeling_kda.py:274-276`). All are dense small tensors, no biases; `nn.Linear(D, 1)`
weight has shape `[1, D]`; `nn.RMSNorm(D)` weight has shape `[D]` (weight only, no
bias). Checkpoint keys under `KDAForCausalLM` (base prefix `model`,
`modeling_kda.py:190,379`):

| checkpoint key | shape | role |
|---|---|---|
| `model.layers.{i}.attn_res_proj.weight` | `[1, D]` | pseudo-query for the attn-side mixture (`modeling_kda.py:136`) |
| `model.layers.{i}.attn_res_norm.weight` | `[D]` | key-RMSNorm scale, attn side (`modeling_kda.py:138`) |
| `model.layers.{i}.mlp_res_proj.weight` | `[1, D]` | pseudo-query, mlp side (`modeling_kda.py:162`) |
| `model.layers.{i}.mlp_res_norm.weight` | `[D]` | key-RMSNorm scale, mlp side (`modeling_kda.py:164`) |
| `model.res_proj.weight` | `[1, D]` | pseudo-query, top-level aggregation (`modeling_kda.py:351`) |
| `model.res_norm.weight` | `[D]` | key-RMSNorm scale, top level (`modeling_kda.py:353`) |

Existing tensors that participate with a **changed role** (used as the folded
`output_rms_weight` instead of a standalone prenorm): `model.layers.{i}.attn_norm.weight`
(`modeling_kda.py:139`), `model.layers.{i}.mlp_norm.weight` (`modeling_kda.py:165`),
`model.norm.weight` (`modeling_kda.py:354`). Same shapes as today; the GGUF converter
doesn't need to touch them, only the graph does.

Net new per layer: `2·(D + D)` params; +`2D` at model level. `rms_eps` = `norm_eps`
everywhere (`modeling_kda.py:91,140`); logit `scale` is the default **1.0** in every
model call (no override passed at `modeling_kda.py:135-141,161-167,350-356`).

---

## 5. Mapping to a ggml graph

Everything composes from existing ops; no new kernel is required for a day-one
correct path. Per attnres call with sources `S_0 … S_{L−1}` (each `[n_embd, n_tokens]`
in ggml layout) and per-call weights `q [D]`, `w_k [D]`, `w_out [D]`:

```
qw    = precompute at load: q ⊙ w_k                     // fused.py:65 identity, weights-only
for i in 0..L-1:
    k_i = ggml_rms_norm(S_i, eps)                        // unit-scale norm; naive.py:67
    s_i = ggml_mul_mat(qw_row, k_i)                      // [1, n_tokens] logits; naive.py:68
logits = ggml_concat(s_0 … s_{L-1})   // dim 0 → [L, n_tokens]
p      = ggml_soft_max(logits)        // softmax over ne0 = depth axis; scale==1.0 so no ggml_scale
o      = Σ_i ggml_mul(S_i, view_row_i(p))               // row view [1, n_tokens] broadcasts over ne0
out    = ggml_mul(ggml_rms_norm(o, eps), w_out)          // folded prenorm; naive.py:72-73
```

Notes, all load-bearing:

- `ggml_soft_max` normalizes over `ne0`, so logits must be laid out with **L along
  ne0** — the concat of `[1, n_tokens]` rows along dim 0 does exactly that.
- `view_row_i(p)` = `ggml_view_2d(p, 1, n_tokens, p->nb[1], i·p->nb[0])`;
  `ggml_mul([D,n_tokens], [1,n_tokens])` broadcasts since `ne0 % 1 == 0`.
- Because `w_k` enters the logit only through `q ⊙ w_k` (`fused.py:65,91`), the
  key-norm weight never needs to be materialized in the graph — bake `qw` per sub-layer
  at load (or keep both tensors and one extra `ggml_mul`; cosmetic choice for the
  converter).
- **Where it attaches in the graph**: replace every
  `cur = build_norm(inpL, attn_norm, …)` read with the block above, and replace the
  plain `inpL = ggml_add(inpL, cur)` stream carry with the prefix-sum discipline of §2
  (carry the running sum; snapshot it into a persistent list at boundaries). The
  final `build_norm(cur, output_norm, …)` before the head is likewise replaced by a
  top-level instance.
- **Intermediate storage across layers**: the snapshot tensors must stay live from
  their boundary until the last consumer (the top-level mixture) — in ggml terms they
  are ordinary graph nodes, but the allocator will hold
  `(num_sources_max)·n_embd·n_tokens·4` bytes of activations concurrently. For block
  mode this is small (`num_blocks + 2` tensors); for full mode it is
  `2·num_layers + 1` tensors **and** the node count becomes O(num_layers²)
  (≈ `4L+2` nodes per call, L growing per sub-layer) — if K3 ships full mode, graph
  node budget and a fused op (modeled on `fused.py:38-121`) become real concerns rather
  than nice-to-haves.
- Precision: the reference does everything in fp32 with one final downcast
  (`naive.py:65-66,75`); ggml `rms_norm`/`soft_max` already accumulate in f32, so the
  composed graph matches without special handling.

---

## 6. Prefill vs decode — what persists

**In fla's reference wiring, nothing persists across forward passes** — whether K3
behaves the same is checkpoint-dependent. `attnres_states` is created as a local
`None` at the top of every `KDAModel.forward` (`modeling_kda.py:325`) and threaded
through the layer loop only (`modeling_kda.py:333-341`); it is never written to
`past_key_values`, and the `Cache` object is untouched by any attnres code (grep clean
outside the two model files and the op). The softmax runs over the **depth** axis per
token (`naive.py:68`) — there is no cross-position interaction, so decoding a token
needs no history of previous tokens' mixtures.

Consequences for llama.cpp — all conditional on K3 following fla's wiring:

- **No memory-module changes.** KV cache / recurrent-state handling
  (`llama_memory_hybrid`, recon 01 §4) is unaffected; AttnRes adds zero cached state.
- **Prefill and decode use the identical graph**, differing only in `n_tokens`. Decode
  is the `n_tokens = 1` degenerate case: L dot products of size D, an L-way softmax,
  and an L-term weighted sum per sub-layer — negligible FLOPs, but the per-sub-layer
  *within-pass* liveness of the snapshot tensors is unavoidable in both modes.
- What must persist **within** one forward pass (across layers, not across tokens):
  the snapshot list + running prefix sum, exactly §2. That is the only "state".

---

## 7. Cross-check of the claimed equation `h_l = Σ_{i<l} α_{i→l} v_i`

The external analysis' form is **directionally right and wrong in four specifics**:

1. **Confirmed**: the sub-layer input is a weighted sum over strictly-earlier
   representations — sources at sub-layer `l` are the embedding plus material from
   sub-layers `< l` only (§2 trace); the current sub-layer's own output is added
   *after* the read (`modeling_kda.py:156,176-179`). So `i < l` is correct.
2. **`α` is not free-form**: `α_{i→l} = softmax_i( q_l · RMSNorm(v_i) )` — a depth
   softmax (Σ_i α = 1, convex), computed **per token**, with a single learned
   pseudo-query vector `q_l` per sub-layer and a shared key-norm (`naive.py:67-68`).
   Writing it as unconstrained per-pair weights `α_{i→l}` overstates the
   parameterization: there is no `[L×L]` learned matrix; zero-init makes it uniform
   (`modeling_kda.py:219-222`).
3. **`v_i` is only "per-sub-layer output" in full mode** (`attnres_block_size = 1`).
   In block mode (even N) the `v_i` are **disjoint block partial sums** plus the
   running intra-block sum, and `v_0` is the token embedding (§3). The equation as
   written silently assumes full mode.
4. **It describes the read, not the stream**: `h_l` in the equation is the sub-layer
   *input* (pre-attn / pre-mlp, then RMS-normed via the folded prenorm). The residual
   stream itself is still built by plain unweighted addition
   (`modeling_kda.py:156,179`) — AttnRes never rewrites the carried sum. The one place
   a mixture *is* the output is the top-level aggregation feeding the LM head
   (`modeling_kda.py:346-356`), which the compact equation doesn't capture.

---

## Open questions (answerable only with the K3 checkpoint / config, announced for July 27)

1. **Is K3's mechanism fla's attnres at all?** `vendor/kimi-linear` has zero attnres
   code, so the only ground truth is fla's op + the KDA/ABC model wiring. If K3's
   "Attention Residuals" differ (e.g. values normalized before mixing, no folded
   prenorm, learned per-source temperature), every section above needs re-verification
   against their modeling code.
2. **`attnres_block_size` for K3** — full mode (`1`), block mode (which even `N`?), and
   is the top-level aggregation present? This decides L, the activation-liveness
   budget, and whether a fused ggml op is needed on day one (§5, full-mode O(layers²)).
3. **Tensor names and granularity in the released checkpoint** — does Moonshot use
   fla's names (`attn_res_proj` / `mlp_res_proj` / `res_proj`, shapes `[1, D]` + `[D]`
   per §4) or their own? Is the pseudo-query per sub-layer (fla) or per layer? This
   defines the GGUF tensor-mapping and arch enums (recon 01 §5 items #5/#7/#9).
4. **Logit scale** — fla's model path uses `scale = 1.0` (no override passed,
   `modeling_kda.py:135-141`); does K3's config set e.g. `D^-1/2` (the op supports it,
   `fused.py:534-535`)?
5. **Do the periodic full-attention (MLA — recon 05 A5) layers also carry AttnRes?** In fla the flag
   is block-level and independent of the attn type (`modeling_kda.py:88`, applies to
   both `Attention` and `KimiDeltaAttention` layers) — confirm K3 doesn't exempt its
   global layers.
6. **Norm-fold layout** — fla reuses `attn_norm`/`mlp_norm`/`norm` weights as the
   folded output norm (§2). Does K3 keep those as the same single prenorm tensors, or
   ship separate dedicated post-mix norms? Determines whether the converter maps 3 or 4
   norm tensors per layer.
7. **Are the pseudo-queries actually non-zero in the trained checkpoint?** Zero-init =
   uniform softmax = vanilla transformer (§2). A quick tensor-norm check on the K3
   weights confirms the mechanism is live (and gives a sanity target for end-to-end
   numeric validation, recon 01 §5 item #10).
8. **`rms_eps`** — fla shares `norm_eps` across key-norm and folded norm
   (`modeling_kda.py:91,140`); confirm K3's value and sharing.
