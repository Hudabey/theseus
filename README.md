# theseus

Analysis and preparation for day-one **Kimi K3** support in the llama.cpp ecosystem.
K3 (announced 2026-07-15 by Moonshot AI) is a 2.8T-parameter MoE with Kimi Delta
Attention (KDA), Attention Residuals, periodic MLA layers, and native-MXFP4 release
weights. Weights are announced for **July 27, 2026**; findings here will be updated
against the released checkpoint.

Everything in these docs is verified against actual code with file:line references —
nothing is asserted from memory or from announcement copy. See
[REFERENCES.md](REFERENCES.md) for the exact upstream commits the file:line citations
resolve against.

## The docs

- **[recon/01-kda-gap-analysis.md](recon/01-kda-gap-analysis.md)** — KDA vs the
  gated-deltanet support already in llama.cpp: the recurrence side-by-side, what
  mainline already ships (more than commonly assumed), and the real gap list for K3.
- **[recon/04-attnres-analysis.md](recon/04-attnres-analysis.md)** — Attention
  Residuals ground truth from the flash-linear-attention reference implementation:
  exact forward math, block semantics, checkpoint tensors, prefill/decode behavior,
  and a mapping onto existing ggml ops.
- **[recon/05-vllm-preview-facts.md](recon/05-vllm-preview-facts.md)** — a fact
  filter for the vLLM K3 preview post: every concrete claim tagged **stated** /
  **derived** / **speculation**, with verbatim quotes. If you are evaluating
  third-party K3 analyses, start here — several widely-repeated "facts" (exact layer
  splits, SiTU definitions) are not in the post.
- **[recon/06-drop-day-runbook.md](recon/06-drop-day-runbook.md)** — the hour-zero
  checklist for the weights release: config and modeling code first, full tensor
  inventory from shard headers via HTTP range requests (no weight download), targeted
  small-tensor pulls, and a decision tree mapping each finding to the work it
  unblocks. Rehearsed end-to-end against Kimi-Linear-48B.

## Status

Pre-release. The runbook is frozen; docs 01/04/05 are current as of 2026-07-22.
After the weights drop, verified findings will be recorded and the docs
corrected where the checkpoint contradicts them.
