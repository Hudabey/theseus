"""theseus convert — Kimi K3 HF checkpoint -> GGUF, MXFP4 preserved losslessly.

Kimi K3 (moonshotai/Kimi-K3) ships its routed experts in native MXFP4
(compressed-tensors `mxfp4-pack-quantized`: per-tensor `weight_packed` U8
[rows, cols/2] + `weight_scale` U8 E8M0 [rows, cols/32], group 32). This
converter never dequantizes them: expert bytes are re-paired into ggml's
`block_mxfp4` layout (a pure byte permutation — nibble reorder + scale-byte
relocation, see `theseus/repack_mxfp4.py`) and written with
`raw_dtype=MXFP4`. Scale bytes pass through unchanged. The emitted file is
byte-provably the checkpoint Moonshot published, not a requantization.

Everything else (attention, shared experts, latent sandwich, router, norms,
embeddings) is BF16/F32 in the release and is written at source precision.

The conversion machinery is vendored from llama.cpp @ 1a064ab with theseus's
`kimi-k3` architecture patch applied — see `theseus/_vendor/PATCHES.md` for
the exact GGUF metadata contract (arch name, tensor names, KV keys). This
module stays structurally close to the vendored `conversion/kimi_linear.py`
(K3's closest shipping relative) and to the two MXFP4-passthrough precedents
`conversion/gpt_oss.py` and `conversion/deepseek.py`.

Every architecture constant asserted here was read from the released
config.json and modeling code — nothing is trusted from documentation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import numpy as np

# --- import bootstrap -------------------------------------------------------
# Make the vendored `gguf` and `conversion` packages importable. Insert at the
# front so a system-installed gguf never shadows the patched one (the patch is
# what knows the kimi-k3 arch).
_VENDOR = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import torch  # noqa: E402

if TYPE_CHECKING:
    from torch import Tensor

from conversion.base import ModelBase, TextModel, gguf, logger  # noqa: E402
from conversion.base import LazyTorchTensor                     # noqa: E402
from conversion.qwen import QwenModel                           # noqa: E402

from theseus.repack_mxfp4 import repack_hf_to_ggml              # noqa: E402


# === Kimi K3 release facts ==================================================
# All values MEASURED from moonshotai/Kimi-K3 (config.json text_config +
# modeling_kimi_linear.py) on 2026-07-27/29. `--allow-nonrelease-shapes`
# (test fixtures) skips the equality asserts; the config keys must still
# exist and are always the values written to the GGUF.
K3_HF_ARCHITECTURES = ("KimiK3ForConditionalGeneration",)

K3_RELEASE = {
    "hidden_size":                  7168,
    "num_hidden_layers":            93,
    "num_experts":                  896,
    "num_experts_per_token":        16,
    "num_shared_experts":           2,
    "moe_intermediate_size":        3072,
    "routed_expert_hidden_size":    3584,      # Stable LatentMoE latent dim
    "routed_scaling_factor":        1.0,
    "first_k_dense_replace":        1,
    "vocab_size":                   163840,
    "max_position_embeddings":      1_048_576,
    "attn_res_block_size":          12,
    "activation_situ_beta":         4.0,
    "activation_situ_linear_beta":  25.0,
    "rms_norm_eps":                 1e-5,
}
# linear_attn_config keys (checked separately — nested)
K3_RELEASE_KDA = {
    "head_dim":                 128,
    "short_conv_kernel_size":   4,
    "gate_lower_bound":         -5.0,
    # 1-based; 24 MLA ({4,8,...,92} ∪ {93}) / 69 KDA
    "full_attn_layers":         list(range(4, 93, 4)) + [93],
}

# VL wrapper prefix on every text tensor; vision is out of scope for the
# text GGUF (a vision/mmproj artifact is a separate concern).
K3_TEXT_PREFIX   = "language_model."
K3_SKIP_PREFIXES = ("vision_tower.", "mm_projector.")

# Routed experts: compressed-tensors MXFP4 pair naming (the ONLY quantized
# tensors in the release — 92 layers x 896 experts x w1/w2/w3 x 2 = 494,592
# U8 tensors; verified against the released shard headers).
K3_EXPERT_QUANT_RE = re.compile(
    r"model\.layers\.(?P<bid>\d+)\.block_sparse_moe\.experts\.(?P<xid>\d+)\."
    r"(?P<wid>w[123])\.weight_(?P<kind>packed|scale)$"
)
# w1: gate, w2: down, w3: up (same convention as kimi_linear.py:193-196)
K3_EXPERT_TENSOR_KEYS = {
    "w1": gguf.MODEL_TENSOR.FFN_GATE_EXP,
    "w2": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
    "w3": gguf.MODEL_TENSOR.FFN_UP_EXP,
}

# AttnRes: per-sublayer proj [1,H] + norm [H] on all 93 layers, plus one
# model-level pair applied before the final norm (modeling_kimi_linear.py
# :910-917, :1215-1233). Names registered in the patched gguf-py
# (constants.py MODEL_TENSOR.*_RES_*); tiny tensors, never quantized.
K3_ATTNRES_TENSOR_MAP = {
    "self_attention_res_proj.weight": "blk.{bid}.attn_res_proj.weight",
    "self_attention_res_norm.weight": "blk.{bid}.attn_res_norm.weight",
    "mlp_res_proj.weight":            "blk.{bid}.ffn_res_proj.weight",
    "mlp_res_norm.weight":            "blk.{bid}.ffn_res_norm.weight",
}
K3_ATTNRES_TOPLEVEL_MAP = {
    "model.output_attn_res_proj.weight": "res_proj.weight",
    "model.output_attn_res_norm.weight": "res_norm.weight",
}

# Stable LatentMoE sandwich (BF16, per MoE layer): shared down/up projections
# around the 896-expert latent space, RMSNorm after expert aggregation, the
# router gate, and the DSv3-style F32 selection-bias vector.
K3_LATENT_MOE_MAP = {
    "block_sparse_moe.routed_expert_down_proj.weight": "blk.{bid}.ffn_latent_down.weight",
    "block_sparse_moe.routed_expert_up_proj.weight":   "blk.{bid}.ffn_latent_up.weight",
    "block_sparse_moe.routed_expert_norm.weight":      "blk.{bid}.ffn_latent_norm.weight",
    "block_sparse_moe.gate.weight":                    "blk.{bid}.ffn_gate_inp.weight",
    "block_sparse_moe.gate.e_score_correction_bias":   "blk.{bid}.exp_probs_b.bias",
}

# `self_attn.g_proj` [12288,7168] BF16 exists on ALL 93 layers: sigmoid
# output gate before o_proj — full-rank KDA gate (use_full_rank_gate) on KDA
# layers, MLA output gate on MLA layers. One GGUF slot (ATTN_GATE); the
# runtime graph disambiguates by layer type.
K3_ATTN_GATE_SUFFIX    = "self_attn.g_proj.weight"
K3_ATTN_GATE_GGUF_NAME = "blk.{bid}.attn_gate.weight"


def _is_k3_mxfp4(quant_config: dict | None) -> bool:
    return (
        isinstance(quant_config, dict)
        and quant_config.get("quant_method") == "compressed-tensors"
        and quant_config.get("format") == "mxfp4-pack-quantized"
    )


@ModelBase.register(*K3_HF_ARCHITECTURES)
class KimiK3Model(TextModel):
    """Kimi K3: KDA + AttnRes + periodic gated MLA + Stable LatentMoE, native MXFP4."""

    model_arch = gguf.MODEL_ARCH.KIMI_K3

    # Release-shape asserts on by default; test fixtures with shrunk dims set
    # this False on the instance before write().
    expect_release: bool = True

    _experts: list[dict[str, Tensor]] | None = None          # float fallback path
    _experts_quant: list[dict[str, Tensor]] | None = None    # MXFP4 packed/scale pairs

    # --- vocab --------------------------------------------------------------
    # K2-lineage tiktoken (clone of kimi_linear.py:22-75). K3 retokenized
    # several special tokens vs K2.5 but keeps the tiktoken model format; the
    # pre-tokenizer identity is asserted, not assumed.
    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
            return
        except Exception:
            pass

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokpre = self.get_vocab_base_pre(tokenizer)

        if tokpre == "kimi-k2":
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.model._mergeable_ranks
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))
            vocab_size = self.hparams["vocab_size"]
            special_tokens = tokenizer.special_tokens
            reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []

            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
            # The GENERATION eos is <|end_of_msg|> (163586, generation_config
            # .json), not the tokenizer's [EOS] (163585) — writing the wrong
            # one produces a model that never stops in chat.
            eos_id = tokenizer.eos_id
            gen_cfg_path = self.dir_model / "generation_config.json"
            if gen_cfg_path.is_file():
                cfg_eos = json.loads(gen_cfg_path.read_text()).get("eos_token_id")
                if isinstance(cfg_eos, int):
                    eos_id = cfg_eos
            self.gguf_writer.add_eos_token_id(eos_id)
        else:
            raise NotImplementedError(f"K3 pre-tokenizer {tokpre!r} is not supported yet")

    # --- release-shape gate --------------------------------------------------
    def _assert_release(self):
        problems = []
        for key, want in K3_RELEASE.items():
            got = self.hparams.get(key)
            if got != want:
                problems.append(f"{key}: config has {got!r}, release is {want!r}")
        lac = self.hparams.get("linear_attn_config") or {}
        for key, want in K3_RELEASE_KDA.items():
            got = lac.get(key)
            if got != want:
                problems.append(f"linear_attn_config.{key}: config has {got!r}, release is {want!r}")
        if not _is_k3_mxfp4(self.hparams.get("quantization_config")):
            problems.append("quantization_config is not compressed-tensors mxfp4-pack-quantized")
        if problems:
            raise ValueError(
                "checkpoint does not match the audited Kimi-K3 release "
                "(pass --allow-nonrelease-shapes only for test fixtures):\n  "
                + "\n  ".join(problems)
            )

    # --- metadata -----------------------------------------------------------
    def set_gguf_parameters(self):
        # (TextModel.__init__ already merged text_config into hparams root.)
        if self.expect_release:
            self._assert_release()

        # MLA -> MQA trick, unchanged from kimi_linear.py:77-78: MLA KV cache
        # is stored in compressed-latent (MQA) form, so head_count_kv is 1 on
        # full-attention layers.
        self.hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        # Hybrid layout: per-layer head_count_kv, 0 = KDA (recurrent),
        # 1 = MLA. The config's full_attn_layers list is 1-based.
        linear_attn_config = self.hparams["linear_attn_config"]
        n_layer = self.hparams["num_hidden_layers"]
        _full_attn_layers = linear_attn_config["full_attn_layers"]
        _num_kv_heads = [
            self.hparams["num_key_value_heads"] if il + 1 in _full_attn_layers else 0
            for il in range(n_layer)
        ]
        assert len(_num_kv_heads) == n_layer
        n_mla = sum(1 for h in _num_kv_heads if h > 0)
        logger.info(f"layer schedule: {n_mla} MLA / {n_layer - n_mla} KDA")
        self.gguf_writer.add_head_count_kv(_num_kv_heads)

        # KDA numerics
        self.gguf_writer.add_ssm_conv_kernel(linear_attn_config["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(linear_attn_config["head_dim"])
        self.gguf_writer.add_kda_gate_lower_bound(linear_attn_config["gate_lower_bound"])

        # MLA params (MQA-form cache), clone of kimi_linear.py:104-132
        if (q_lora_rank := self.find_hparam(["q_lora_rank", "n_lora_q"], optional=True)) is not None:
            self.gguf_writer.add_q_lora_rank(q_lora_rank)
        kv_lora_rank = self.find_hparam(["kv_lora_rank", "n_lora_kv"], optional=False)
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)

        qk_nope_head_dim = self.hparams.get("qk_nope_head_dim")
        qk_rope_head_dim = self.find_hparam(["qk_rope_head_dim", "n_rot"], optional=False)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        v_head_dim = self.hparams.get("v_head_dim")

        if (n_embd_head_k_mla := self.find_hparam(["n_embd_head_k_mla"], optional=True)) is not None:
            self.gguf_writer.add_key_length_mla(n_embd_head_k_mla)
        elif qk_nope_head_dim is not None:
            self.gguf_writer.add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)

        if (n_embd_head_v_mla := self.hparams.get("n_embd_head_v_mla")) is not None:
            self.gguf_writer.add_value_length_mla(n_embd_head_v_mla)
        elif v_head_dim is not None:
            self.gguf_writer.add_value_length_mla(v_head_dim)

        # MoE. super() already emitted expert_count/expert_used_count from
        # num_experts/num_experts_per_token.
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(self.hparams["num_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])

        # Kimi K3 specifics — the KV contract in _vendor/PATCHES.md. Values
        # come from the checkpoint config, never from constants.
        self.gguf_writer.add_latent_moe_dim(self.hparams["routed_expert_hidden_size"])
        self.gguf_writer.add_attnres_block_size(self.hparams["attn_res_block_size"])
        self.gguf_writer.add_situ_beta(self.hparams["activation_situ_beta"])
        self.gguf_writer.add_situ_linear_beta(self.hparams["activation_situ_linear_beta"])

    # --- tensors ------------------------------------------------------------
    def dequant_model(self):
        # K3's routed experts are compressed-tensors MXFP4. The generic
        # dequant path has no branch for `mxfp4-pack-quantized` (it would
        # raise), and it must never run anyway: the pairs are consumed
        # losslessly in modify_tensors. Precedent: gpt_oss.py:18-21,
        # deepseek.py selective dequant.
        if _is_k3_mxfp4(self.hparams.get("quantization_config")):
            self._is_mxfp4 = True
            return
        if self._is_mxfp4:
            return
        return super().dequant_model()

    def prepare_tensors(self):
        super().prepare_tensors()
        for bufs, what in ((self._experts, "float experts"), (self._experts_quant, "quant expert pairs")):
            if bufs is not None:
                leftover = [k for d in bufs for k in d.keys()]
                if len(leftover) > 0:
                    raise ValueError(f"Unprocessed {what}: {leftover}")
        if self._experts_quant is not None:
            # MXFP4 pairs were consumed -> the file is MXFP4-MoE regardless of
            # the skeleton outtype (deepseek.py:773-776 precedent).
            self._is_mxfp4 = True
            self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Consume the MXFP4 expert pairs at RAW dtype and write them repacked.

        This runs before the main tensor loop, which lazily casts every
        non-f16/f32 tensor to float32 before modify_tensors — a cast that
        would destroy U8 packed/scale pairs. gpt_oss.py:63-81 is the shipping
        precedent for this structure; modify_tensors then drops the pairs
        when the main loop reaches them.
        """
        if not _is_k3_mxfp4(self.hparams.get("quantization_config")):
            return []
        n_experts = self.find_hparam(["num_experts", "num_local_experts", "n_routed_experts"])
        self._experts_quant = [{} for _ in range(self.block_count)]
        for name, data_torch in self.get_tensors():
            if name.startswith(K3_TEXT_PREFIX):
                name = name[len(K3_TEXT_PREFIX):]
            m = K3_EXPERT_QUANT_RE.fullmatch(name)
            if m is None:
                continue
            bid = int(m["bid"])
            self._experts_quant[bid][name] = data_torch
            # 3 projections x (packed + scale) per expert
            if len(self._experts_quant[bid]) >= n_experts * 3 * 2:
                for wid in ("w1", "w2", "w3"):
                    self._emit_mxfp4_expert_stack(bid, wid, n_experts)
        return []

    def _emit_mxfp4_expert_stack(self, bid: int, wid: str, n_experts: int):
        """Stack one layer's per-expert MXFP4 pairs into a 3D ggml MXFP4 tensor.

        Pure passthrough repack — no dequant/requant. Byte shuffle:
        theseus/repack_mxfp4.py, bit-exactness proven in
        tests/test_repack_mxfp4.py and re-checkable on any emitted file with
        `theseus verify`. Vendor precedent for raw add_tensor of stacked
        expert bytes: deepseek.py:624-649, gpt_oss.py:48-61.
        """
        assert self._experts_quant is not None
        data: np.ndarray | None = None
        for xid in range(n_experts):
            base = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}"
            blocks = LazyTorchTensor.to_eager(self._experts_quant[bid].pop(base + ".weight_packed"))
            scales = LazyTorchTensor.to_eager(self._experts_quant[bid].pop(base + ".weight_scale"))
            raw = repack_hf_to_ggml(
                blocks.contiguous().view(torch.uint8).numpy(),
                scales.contiguous().view(torch.uint8).numpy(),
            )  # [rows, n_blocks*17]
            if data is None:
                data = np.empty((n_experts, *raw.shape), dtype=raw.dtype)
            data[xid] = raw
        assert data is not None
        new_name = self.format_tensor_name(K3_EXPERT_TENSOR_KEYS[wid], bid)
        shape = gguf.quant_shape_from_byte_shape(data.shape, gguf.GGMLQuantizationType.MXFP4)
        logger.info(f"{new_name}: MXFP4 passthrough, logical shape = {shape}")
        self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # VL wrapper: text tensors carry the language_model. prefix; vision is
        # out of scope for the text GGUF
        if name.startswith(K3_SKIP_PREFIXES):
            return
        if name.startswith(K3_TEXT_PREFIX):
            name = name[len(K3_TEXT_PREFIX):]

        # LatentMoE sandwich + router — before the expert paths so the generic
        # experts regex never swallows them
        for suffix, out_fmt in K3_LATENT_MOE_MAP.items():
            if name.endswith("." + suffix):
                assert bid is not None
                yield (out_fmt.format(bid=bid), data_torch)
                return

        # AttnRes tensors — tiny [1,H]/[H], written at source precision
        for suffix, out_fmt in K3_ATTNRES_TENSOR_MAP.items():
            if name.endswith("." + suffix):
                assert bid is not None
                yield (out_fmt.format(bid=bid), data_torch)
                return
        if name in K3_ATTNRES_TOPLEVEL_MAP:
            yield (K3_ATTNRES_TOPLEVEL_MAP[name], data_torch)
            return

        # Output gate (all 93 layers; KDA full-rank gate / MLA output gate
        # share the tensor slot)
        if name.endswith(K3_ATTN_GATE_SUFFIX):
            assert bid is not None
            yield (K3_ATTN_GATE_GGUF_NAME.format(bid=bid), data_torch)
            return

        # KDA conv1d reshape — clone of kimi_linear.py:153-170. HF
        # [d_inner, d_conv] -> ggml ne [d_conv, 1, d_inner, 1]; layout already
        # matches, shape-only.
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)
            elif data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # KDA decay: -exp(A_log) baked at conversion (kimi_linear.py:172-179)
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # Routed experts, MXFP4 passthrough path: already consumed at raw
        # dtype in generate_extra_tensors — drop the main loop's f32-cast copy
        if K3_EXPERT_QUANT_RE.fullmatch(name) is not None:
            return

        # Routed experts, float fallback (unquantized checkpoints — never the
        # release; kept so a future bf16 drop converts unchanged)
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.find_hparam(["num_experts", "num_local_experts", "n_routed_experts"])
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for wid, tname in [("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                   ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                   ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    stacked = torch.stack(datas, dim=0)
                    new_name = self.format_tensor_name(tname, bid)
                    yield from super().modify_tensors(stacked, new_name, bid)
            return

        # MLA kv_b split for the absorption optimization (kimi_linear.py
        # :207-221; k_b transposed for the absorbed matmul)
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")
            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.find_hparam(["n_embd_head_v_mla", "v_head_dim"], optional=False)
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)
            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)


# --- CLI --------------------------------------------------------------------

def _parse_size(text: str) -> int:
    m = re.fullmatch(r"(\d+)([KMG]?)", text.strip(), flags=re.IGNORECASE)
    if m is None:
        raise argparse.ArgumentTypeError(f"invalid size {text!r} (expected e.g. 45G)")
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m[2].upper()]
    return int(m[1]) * mult


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="theseus convert",
        description="Convert a Kimi K3 HF checkpoint to GGUF with native MXFP4 preserved byte-exactly.",
    )
    ap.add_argument("model_dir", type=Path, help="local directory containing the downloaded HF checkpoint")
    ap.add_argument("--outfile", type=Path, default=None,
                    help="output path or directory (default: <model_dir>/gguf/); "
                         "{ftype} in the name is templated")
    ap.add_argument("--split-max-size", type=_parse_size, default=_parse_size("45G"),
                    help="max size per GGUF split (default 45G, under the HF 50GB limit); 0 = no split")
    ap.add_argument("--dry-run", action="store_true", help="plan the conversion without writing tensor data")
    ap.add_argument("--allow-nonrelease-shapes", action="store_true",
                    help="skip the release-shape asserts (test fixtures only)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not (args.model_dir / "config.json").is_file():
        logger.error(f"no config.json under {args.model_dir} — pass the downloaded checkpoint directory")
        return 2

    out = args.outfile if args.outfile is not None else args.model_dir / "gguf"
    out.mkdir(parents=True, exist_ok=True) if out.suffix == "" else out.parent.mkdir(parents=True, exist_ok=True)

    model = KimiK3Model(
        args.model_dir,
        gguf.LlamaFileType.MOSTLY_BF16,   # skeleton precision; flips to MXFP4_MOE when expert pairs are consumed
        out,
        split_max_tensors=0,
        split_max_size=args.split_max_size,
        dry_run=args.dry_run,
    )
    model.expect_release = not args.allow_nonrelease_shapes
    model.write()
    logger.info(f"done: {model.fname_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
