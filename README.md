# Theseus: Kimi K3 support reconnaissance for llama.cpp

Pre-release implementation analysis and release-day verification tooling for
**potential Kimi K3 support in llama.cpp. No working K3 port exists yet.**

K3 (announced 2026-07-15 by Moonshot AI) is a 2.8T-parameter MoE with Kimi Delta
Attention (KDA), Attention Residuals, periodic MLA layers, and native-MXFP4 release
weights, announced for release by **July 27, 2026**. Findings here will be updated
against the released checkpoint.

Implementation claims are tied to commit-pinned code references. Pre-release K3
claims are separately labeled as externally stated, derived, or unresolved. See
[REFERENCES.md](REFERENCES.md) for the exact upstream commits the citations resolve
against.

## Status at a glance

| basis | items |
|---|---|
| **Confirmed in current llama.cpp** (code-verified) | `ggml_gated_delta_net` with a per-channel (KDA) gate mode on all backends; a complete `LLM_ARCH_KIMI_LINEAR` KDA graph; a Kimi-Linear converter; `GGML_TYPE_MXFP4` as a loadable type. (recon/01 §3) |
| **Stated by vLLM** (quoted, not independently verifiable yet) | 2.8T params; 93 layers; MLA every four layers with a new gate projection; 896 routed experts / 16 active + shared; "SiTU" MoE activation (undefined); "Stable LatentMoE" (unexplained); MXFP4 release weights. (recon/05) |
| **Derived working hypothesis** (plausible, unproven) | K3's AttnRes ≈ fla's block-mode `attnres` op; KDA numerics match Kimi-Linear; ≈3:1 KDA:MLA layout; HF MXFP4 serialization repacks losslessly into `block_mxfp4`. (recon/04, recon/05 [D] tags) |
| **Requires K3 release** | SiTU formula and parameters; LatentMoE structure; AttnRes block size/tensors; layer-placement offset; MLA gate wiring; MXFP4 packing confirmation; tokenizer/template. (recon/06 findings checklist) |

## The docs

- **[recon/01-kda-gap-analysis.md](recon/01-kda-gap-analysis.md)** — KDA vs the
  gated-deltanet support already in llama.cpp: the recurrence side-by-side, what
  mainline already ships (more than commonly assumed), and the real gap list for K3.
- **[recon/02-mxfp4-preservation.md](recon/02-mxfp4-preservation.md)** — native-MXFP4
  preservation into GGUF: the exact `block_mxfp4` layout, why byte-level passthrough
  is bit-exact, the two shipping converter precedents (gpt-oss, DeepSeek-V4), the
  lossless-repack checklist, and the T1–T4 bit-exactness acceptance gates. The
  repacker itself and its runnable oracle suite are in this repo
  ([src/repack_mxfp4/](src/repack_mxfp4/), [tests/oracle/](tests/oracle/)). Scope
  note: K3's actual MXFP4 serialization is open-question-gated until the checkpoint.
- **[recon/04-attnres-analysis.md](recon/04-attnres-analysis.md)** — Attention
  Residuals ground truth from the flash-linear-attention reference implementation:
  exact forward math, block semantics, checkpoint tensors, prefill/decode behavior,
  and a mapping onto existing ggml ops. Scope note: everything there describes fla's
  wiring; whether K3 follows it is checkpoint-dependent.
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
- **[tools/drop_day/](tools/drop_day/)** — the five standalone scripts the runbook
  invokes: release watcher, upload-completeness gate, shard-header fetch (HTTP range
  requests), tensor-name classifier, and small-tensor puller.
  `pull_small.py --self-test` runs the offline BF16-decode unit test.

## Status

Pre-release. The runbook is frozen; docs 01/04/05 are current as of 2026-07-23.
After the weights drop, verified findings will be recorded and the docs
corrected where the checkpoint contradicts them.

## Prerequisites

Runbook tooling: `pip install -r requirements.txt` (`huggingface_hub`, `safetensors`,
`requests`, `numpy`), plus the `hf` CLI (ships with `huggingface_hub`) and `jq` on
PATH.

Exact versions used in the 2026-07-22 rehearsal: Python 3.12.2, `huggingface_hub`
1.3.5 (also provides the `hf` CLI), `safetensors` 0.5.2, `requests` 2.32.4, `numpy`
1.26.4, `jq` 1.6.

Oracle test suite (MXFP4 repacker bit-exactness, recon/02 §6):
`python -m pytest tests/oracle/test_repack_mxfp4.py`

## License

MIT — see [LICENSE](LICENSE).
