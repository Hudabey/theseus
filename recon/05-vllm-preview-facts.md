# 05 — vLLM K3 preview: verified fact list

Source: https://vllm.ai/blog/2026-07-22-kimi-k3-preview ("A Preview of Production-Scale
Kimi K3 Support on vLLM", published 2026-07-22), fetched raw 2026-07-22. All quotes are
verbatim from the page text. Purpose: this doc is the **filter for claims from other
analyses** — anything an external analysis asserts about K3 that is not in the
"stated" column below is speculation until the July 27 weights drop.

Legend: **[V]** = stated by the vLLM post (quote given). **[D]** = derivable from a [V]
fact by arithmetic/close reading — flagged, since the derivation can be wrong.
**[S]** = *not* in the post; anything claiming it is speculation.

---

## 1. Architecture facts

| # | Fact | Status | Evidence |
|---|---|---|---|
| A1 | 2.8T total parameters | **[V]** | "a 2.8-trillion-parameter model" |
| A2 | 1M-token context window | **[V]** | "a 1-million-token context window" |
| A3 | Native vision, via a vision tower, **image-only** preprocessing | **[V]** | "Native vision with a vision tower — Requires multimodal preprocessing (image-only)" |
| A4 | Attention is hybrid: "KDA-dominant linear attention with periodic full-attention layers" | **[V]** | TL;DR bullet, verbatim |
| A5 | The full-attention layers are **MLA**, every four layers | **[V]** | "Kimi K3 still uses MLA attention every four layers" |
| A6 | MLA has a **gate projection** new in K3, runnable in parallel with the attention path; the gate epilogue is an **elementwise multiply + sigmoid** | **[V]** | "Kimi K3 introduces a gate projection that can execute in parallel with the main attention path … we fuse the elementwise multiply and sigmoid into the gate-projection epilogue" |
| A7 | The network is **93 layers** | **[V]** | "throughout the 93-layer network" (in the AttnRes section) |
| A8 | KDA per-layer state = "a matrix-like recurrent state, together with a short convolution state"; layer includes gating and normalization | **[V]** | prefix-caching section; decode kernel covers "the short convolution, KDA state update, output gate, and normalization" |
| A9 | MoE: **896 routed experts, 16 active per token, plus shared experts** | **[V]** | table row, verbatim |
| A10 | A named component "**Stable LatentMoE**" exists | **[V]** | TL;DR: "KDA-dominant linear attention with periodic full-attention layers, AttnRes across depth, Stable LatentMoE, and native vision support" — named once, **never elaborated** |
| A11 | Release configuration weights are **MXFP4** | **[V]** | "MXFP4 weights in the provided release configuration"; "Kimi K3's release configuration uses MXFP4 weights" |
| A12 | ~23 MLA / ~70 KDA layer split | **[D]** | 93 layers (A7) ÷ every-4 MLA (A5) ⇒ ≈23/70. The post **never states the split**; 93 is not divisible by 4, the placement offset is unknown, and whether "93" counts only decoder layers is unknown. Treat any exact split from other analyses as unconfirmed. |
| A13 | 3:1 KDA:MLA ratio | **[D]** | Follows from "every four layers" (A5); matches Kimi-Linear's published 3:1 (recon 01 §5). Uniformity/offset not stated. |
| A14 | All non-MLA layers are KDA (no other layer type) | **[D]** | Implied by A4 + A8; not stated exhaustively. |
| A15 | Number/size of shared experts, expert dims, top-k grouping params | **[S]** | "plus shared experts" is all the post gives; "optimized grouped top-k routing" names the mechanism, no numbers. |
| A16 | MLA NoPE vs RoPE, head counts, KDA head_dim/chunk size, vocab size, hidden size | **[S]** | none of these appear anywhere in the post. |

## 2. AttnRes facts

| # | Fact | Status | Evidence |
|---|---|---|---|
| R1 | AttnRes reads from **earlier layer blocks**, not a single accumulated stream | **[V]** | "AttnRes retrieves from representations written by earlier layer blocks rather than relying on only one uniformly accumulated residual stream" |
| R2 | It is a depth-axis mechanism with cross-layer reads *and writes* | **[V]** | table row "Depth — Attention Residual — Adds cross-layer representation reads and writes that need dedicated kernels" |
| R3 | The serving kernels fuse "**residual update, AttnRes mixing, and output RMSNorm**" (elsewhere: "fusion of residual addition and output RMSNorm on supported shapes") | **[V]** | AttnRes section + status table |
| R4 | Attention-residual traffic is sharded via sequence parallelism | **[V]** | "Sequence-parallel work also shards the attention-residual traffic across ranks" |
| R5 | K3's AttnRes = fla's `attnres` op (recon 04) | **[D]** | Strongly consistent, not stated: "earlier layer **blocks**" matches fla's block mode (recon 04 §3), and R3's fused triple (residual add + mixing + output RMSNorm) is exactly the shape of fla's fused op with folded `output_rms_weight` (`fused.py:116-119`) plus the prefix-sum write (recon 04 §2). **The post never mentions fla, a block size, pseudo-queries, or a top-level aggregation.** |
| R6 | AttnRes block size, per-sublayer vs per-layer granularity, tensor names/shapes, logit scale, whether MLA layers carry it | **[S]** | absent from the post — recon 04's open questions all remain open. |

## 3. SiTU — the term is real, its meaning is not published

**Confirmed: "SiTU" exists as a term in this post.** Four occurrences, all as an
*activation function in the MoE expert path*:

1. TL;DR: "…**SiTU-enabled MXFP4 MoE execution**, and optimized expert routing."
2. Glance table: "Needs an efficient FP4 MoE path with Kimi K3's **SiTU activation**."
3. NVIDIA path: "Kimi K3's release configuration uses MXFP4 weights and the **SiTU
   activation**. Before this work, the MXFP4 TRTLLM-Gen path did not support SiTU and
   would fall back to a slower implementation. vLLM now maps Kimi K3's **SiTU
   parameters** into the optimized FP4 expert path…"
4. AMD path: FlyDSL stack includes "hardware-tuned A16W4/A8W4 quantized fused operators
   and a **SiTU activation implementation**."

What is **not** in the post: any expansion of the acronym, any formula, any statement
of what "SiTU parameters" are (learned per-expert tensors vs config scalars). **Any
external analysis that expands or defines SiTU is speculating.** Cross-checked locally:
zero hits for `SiTU` in all four vendor repos (`vendor/fla`, `vendor/llama.cpp`,
`vendor/ik_llama.cpp`, `vendor/kimi-linear`; grep 2026-07-22) — it is not an existing
fla/ggml concept under that name.

What we *can* safely take from it: (a) the expert activation is **not** plain
SwiGLU/SiLU, otherwise TRTLLM-Gen would not have "not supported" it; (b) it carries
*parameters* of some kind that a serving engine must map per expert path. Both matter
for the GGUF converter and graph (see §5).

## 4. Serving/infra facts (context for claims, not port targets)

- **Weights release: "by July 27, 2026"** [V]; announcement/weights deliberately
  separated as a process choice [V].
- KDA breaks conventional prefix caching; Moonshot contributed a vLLM implementation
  separating **physical state-block size / scheduler alignment / prefix-match unit**,
  with copy-on-write on partial block hits [V]. (Relevant background: confirms KDA
  state is "much larger than one ordinary token's KV entry" [V].)
- Kernel roster on the release branch [V]: FlashKDA integration (FlashKDA is real and
  pre-existing — Moonshot CUTLASS backend, `vendor/fla/ENVs.md:55`,
  github.com/MoonshotAI/FlashKDA), fused KDA decode (conv + state update + gate +
  norm), fused KDA projections/convolution, fused AttnRes, reimplemented MLA module
  with separate prefill/decode paths, SiTU-enabled MXFP4 via TRTLLM-Gen and DeepGEMM,
  grouped top-k routing; AMD via FlyDSL MLIR (A16W4/A8W4).
- Parsers/semantics implemented in vLLM [V]: "chat rendering, tokenizer integration,
  streaming parsing, tool calls, reasoning output, and structured-output paths" — all
  "under final end-to-end validation". No parser formats/specs given [S: any claim
  about K3's tool-call format].
- Status honesty: much of the post is explicitly "in progress" / "being validated" /
  "final validation loop" — the post itself is a preview, not a spec.

## 5. What this changes for theseus (recon 01/04 cross-refs)

1. **Recon 01 OQ3 answered [V]:** full-attention layers are **MLA** — reuse the
   Kimi-Linear MLA path (`vendor/llama.cpp/src/models/kimi-linear.cpp:374-473`) — but
   A6's **new gate projection (sigmoid ⊙ multiply)** is *not* in Kimi-Linear's MLA and
   is net-new graph work.
2. **Recon 01 OQ4 tightened [V/D]:** layout is every-4-MLA (≈3:1), same as
   Kimi-Linear's published ratio; the signaling mechanism in the checkpoint is still
   unknown.
3. **Recon 04 OQ1/OQ2 tightened [D]:** "earlier layer blocks" + the fused
   residual-add/mix/output-RMSNorm triple match fla's block-mode attnres closely
   enough to keep building on recon 04 as the working model. Block size and tensor
   inventory still gate the converter.
4. **New gap, absent from recon 01's change list: SiTU.** ggml MoE expert activation
   is currently a fixed op choice (SiLU/GELU family); a parametric activation means a
   new ggml op (or graph composition) + new GGUF tensors/metadata + converter support.
   Until defined, budget it as unknown-M/L.
5. **New unknown: "Stable LatentMoE".** If K3's MoE routing/latent structure deviates
   from standard routed-experts (the name suggests it might), recon 01's assumption
   that MoE is conversion-only work is at risk. Nothing more can be said from this
   post.
6. **MXFP4 confirmed as the release format [V]** — recon 01 item #8 (native MXFP4
   passthrough) is confirmed as mandatory, not precautionary. Block layout
   compatibility with `GGML_TYPE_MXFP4` remains unverified.
7. **93 layers [V]** sizes the AttnRes liveness math from recon 04 §5: even in block
   mode the snapshot count stays small; a full-mode 93-layer graph (187 sources at the
   head) would be prohibitive, which weakly supports block mode [D].

---

## Open questions

1. **SiTU**: expansion, formula, and what "SiTU parameters" are (per-expert learned
   tensors → new GGUF tensors; or scalars → metadata). Only the K3 release
   (config/code) can answer; the vLLM post deliberately doesn't.
2. **Stable LatentMoE**: is this branding for standard shared+routed experts, or a
   structurally different (latent/factored) MoE? Determines whether recon 01's MoE
   assumptions hold.
3. **Layer arithmetic**: what exactly is 93 (decoder layers only? incl. vision?), and
   the MLA placement offset given 93 mod 4 ≠ 0.
4. **AttnRes specifics** (block size, tensors, scale, top-level aggregation, MLA-layer
   participation): all still open — recon 04's open-question list stands unchanged.
5. **MLA gate projection**: gating what (attention output? value path?), tensor shape,
   and whether it replaces Kimi-Linear's NoPE-MLA output path or adds to it.
6. **Does "93-layer network" + "MLA attention every four layers" hold in the released
   config**, or was the post written against a pre-freeze checkpoint? The post itself
   warns the model team is still "stabilizing … the final checkpoint, configuration".
