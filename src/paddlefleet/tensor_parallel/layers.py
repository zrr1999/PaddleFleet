# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed.communication.reduce_scatter import _reduce_scatter_base
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    mark_as_sequence_parallel_parameter,
)

from ..parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# from ..dist_checkpointing.mapping import ShardedStateDict
# from ..transformer.utils import make_sharded_tensors_for_checkpoint
from ..utils import (
    divide,
    get_pg_rank,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
    prepare_input_tensors_for_wgrad_compute,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from .utils import VocabUtility

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_deep_gemm_available = True
try:
    from paddlefleet_ops import deep_gemm
except (ImportError, RuntimeError):
    _deep_gemm_available = False
    deep_gemm = None

# Color tag for fp8-enabled non-MoE Linear weights so that
# ``optimizer.clear_param_storage("linear_fp8")`` can free their bf16 master
# storage after pre-quantization. Mirrors the MoE ``"moe_expert"`` pattern.
_LINEAR_FP8_COLOR = "linear_fp8"


def _fp8_prequant_weight(layer):
    """Pre-quantize ``layer.weight`` and stash fp8 cache attrs on it.

    Stashes forward-orientation fp8 tensor + both fwd/bwd scales. The
    backward-orientation fp8 tensor is derived on demand via
    ``fp8_weight_fwd.T.contiguous()`` in ``weight_quant_func``.
    """
    if not getattr(layer, "fp8", False):
        return
    if getattr(layer, "weight_quant_func", None) is None:
        return
    weight = getattr(layer, "weight", None)
    if weight is None:
        return
    prev_fp8 = getattr(weight, "fp8_weight_fwd", None)
    if prev_fp8 is not None:
        try:
            delattr(weight, "fp8_weight_fwd")
        except AttributeError:
            weight.fp8_weight_fwd = None
    _, scale_bwd, fp8_fwd, scale_fwd = layer.weight_quant_func(weight)
    weight.fp8_weight_fwd = fp8_fwd
    weight.fp8_scale_fwd = scale_fwd
    weight.fp8_scale_bwd = scale_bwd


def _fp8_clear_prequant_weight(layer):
    """Drop the fp8 cache stashed by ``_fp8_prequant_weight``.

    Symmetric to ``_fp8_prequant_weight``: strips ``fp8_weight_fwd`` and
    both scale attrs from ``layer.weight`` so the next ``weight_quant_func``
    call re-quantizes the (post-optimizer-step) bf16 weight. Safe to call
    when no cache exists or the layer is bf16.
    """
    weight = getattr(layer, "weight", None)
    if weight is None:
        return
    for attr in ("fp8_weight_fwd", "fp8_scale_fwd", "fp8_scale_bwd"):
        if hasattr(weight, attr):
            try:
                delattr(weight, attr)
            except AttributeError:
                setattr(weight, attr, None)


def _maybe_color_linear_fp8_weight(layer):
    """Tag ``layer.weight`` with the linear-fp8 color, once, if unset."""
    if not getattr(layer, "fp8", False):
        return
    # Expert linears are colored by MoELayer with "moe_expert"; skip here to
    # avoid Paddle's guard against reassigning a non-None color.
    if getattr(layer, "is_expert", False):
        return
    weight = getattr(layer, "weight", None)
    if weight is None:
        return
    color = getattr(weight, "color", None)
    if color in (None, -1):
        weight.color = {"color": _LINEAR_FP8_COLOR}


HAVE_TE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..transformer.transformer_config import TransformerConfig

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def param_is_not_tensor_parallel_duplicate(param):
    """Returns true if the passed-in parameter is not a duplicate parameter
    on another TP rank."""
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    """Sets tp attributes to tensor"""
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    tensor.tensor_model_parallel = is_parallel
    tensor.partition_dim = dim
    tensor.partition_stride = stride


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    """Set default model parallel attributes if not set explicitly already."""

    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    """Copy model parallel attributes from one tensor to another."""

    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(
                destination_tensor, attribute, getattr(source_tensor, attribute)
            )

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, is_expert=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if not is_expert:
        if dist.get_world_size() <= 1:
            init_method(weight)
        else:
            with get_cuda_rng_tracker().fork():
                init_method(weight)
    else:
        if dist.get_world_size() <= 1:
            init_method(weight)
        else:
            with get_cuda_rng_tracker().fork(
                get_expert_parallel_rng_tracker_name()
            ):
                init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    input_size,
    output_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=paddle.float32,
    rank=None,
    world_size=None,
    skip_set_tensor_parallel_attributes=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    if not skip_set_tensor_parallel_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    # Initialize master weight
    master_weight = paddle.empty(
        [input_size, output_size], dtype=paddle.float, requires_grad=False
    )
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)
    # Split and copy

    per_partition_per_stride_size = divide(per_partition_size, stride)
    split_num = divide(
        master_weight.shape[partition_dim], per_partition_per_stride_size
    )
    weight_list = paddle.split(
        master_weight, num_or_sections=split_num, axis=partition_dim
    )
    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with paddle.no_grad():
        # all tensors must live on the same device
        cpu_weight = paddle.cat(my_weight_list, dim=partition_dim)
        weight.copy_(cpu_weight)
    if return_master_weight:
        return master_weight
    return None


_EMBED_IDS_CALL = 0


def _dump_embed_lookup_ids(input_, masked_input, input_mask, vocab_start, vocab_end):
    """Dump-only embedding lookup ids. Observation, not a wrap."""
    dump = os.environ.get("MODEL_REPRO_EMBED_IDS_DIR")
    hashdir = os.environ.get("MODEL_REPRO_EMBED_IDS_HASH_DIR")
    if not dump and not hashdir:
        return
    import hashlib
    import json

    global _EMBED_IDS_CALL
    _EMBED_IDS_CALL += 1
    try:
        rank = int(dist.get_rank()) if dist.is_initialized() else 0
    except Exception:
        rank = 0
    step_env = os.environ.get("MODEL_REPRO_STEP", "")
    ids = masked_input.detach().cpu().numpy()
    raw = input_.detach().cpu().numpy()
    mask = None
    if input_mask is not None:
        mask = input_mask.detach().cpu().numpy().astype("uint8")
    n = int(ids.size)
    prefix = ids.reshape(-1)[:-1] if n > 1 else ids.reshape(-1)
    meta = {
        "kind": "fwd",
        "tag": "embed_ids",
        "framework": "paddle",
        "rank": rank,
        "call": _EMBED_IDS_CALL,
        "step_env": step_env,
        "shape": list(ids.shape),
        "n": n,
        "vocab_start": int(vocab_start),
        "vocab_end": int(vocab_end),
        "masked_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "prefix_n": int(prefix.size),
        "prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
        "n_oov": int(mask.sum()) if mask is not None else 0,
        "n_unique_masked": int(len(set(ids.reshape(-1).tolist()))),
    }
    if hashdir:
        os.makedirs(hashdir, exist_ok=True)
        with open(os.path.join(hashdir, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        if _EMBED_IDS_CALL == 1:
            print(
                f"[E576-EMBED-IDS-HASH] dir={hashdir} rank={rank} "
                f"n={n} prefix_n={meta['prefix_n']}",
                flush=True,
            )
    if dump:
        os.makedirs(dump, exist_ok=True)
        stem = f"paddle_embed_r{rank}_c{_EMBED_IDS_CALL}_L{int(ids.size)}"
        ids.tofile(os.path.join(dump, f"{stem}_masked.i64.bin"))
        raw.tofile(os.path.join(dump, f"{stem}_input.i64.bin"))
        if mask is not None:
            mask.tofile(os.path.join(dump, f"{stem}_mask.u8.bin"))
        with open(os.path.join(dump, f"{stem}.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, sort_keys=True)
            handle.write("\n")
        print(
            f"[EMBED-IDS-DUMP] r{rank} c{_EMBED_IDS_CALL} n={ids.size} "
            f"oov={meta['n_oov']} sha={meta['masked_sha256'][:16]}",
            flush=True,
        )


_EMBED_CHAIN_HITS: dict[str, int] = {}


def _maybe_dump_embed_chain(tensor, tag: str):
    """Dump-only dY of VocabParallelEmbedding allreduce/slice. CPU, n in {168,169}."""
    dump = os.environ.get("MODEL_REPRO_EMBED_CHAIN_DIR")
    if not dump or tensor is None:
        return tensor
    if getattr(tensor, "stop_gradient", True):
        return tensor

    def _hook(g, *, _dump=dump, _tag=tag):
        if g is None:
            return g
        hidden = int(g.shape[-1]) if g.ndim >= 1 else 1
        ntok = int(g.size) // hidden if hidden else 0
        if ntok not in (168, 169):
            return g
        import hashlib
        import json

        try:
            rank = int(dist.get_rank()) if dist.is_initialized() else 0
        except Exception:
            rank = 0
        key = f"{rank}|{_tag}"
        _EMBED_CHAIN_HITS[key] = _EMBED_CHAIN_HITS.get(key, 0) + 1
        hit = _EMBED_CHAIN_HITS[key]
        os.makedirs(_dump, exist_ok=True)
        dy = g.detach().cpu().astype("float32").numpy()
        stem = f"paddle_chain_{_tag}_r{rank}_h{hit}_L{ntok}"
        dy.tofile(os.path.join(_dump, f"{stem}.f32.bin"))
        meta = {
            "framework": "paddle",
            "tag": _tag,
            "rank": rank,
            "hit": int(hit),
            "ntok": ntok,
            "dy_shape": list(dy.shape),
            "dy_sha256": hashlib.sha256(dy.tobytes()).hexdigest(),
        }
        with open(os.path.join(_dump, f"{stem}.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, sort_keys=True)
            handle.write("\n")
        print(
            f"[EMBED-CHAIN] {_tag} r{rank} h={hit} n={ntok} "
            f"shape={tuple(dy.shape)} sha={meta['dy_sha256'][:16]}",
            flush=True,
        )
        return g

    tensor.register_hook(_hook)
    return tensor


class _EmbedFp32MainGrad(paddle.autograd.Function):
    """UAC embedding lookup whose wgrad lands in fp32 main_grad.

    Forward is `weight[ids]` (bf16 activation unchanged). Backward deposits
    via a nested IndexingBackward on an fp32 view of the accumulator
    (iso_samew_fp32 `two`, not embedding_grad_add_to_ which is nz 233424).
    Returns None for weight.grad so MixPrecision cannot add_(bf16).
    """

    _printed = 0
    _dy_hits = 0

    @staticmethod
    def forward(ctx, weight, ids):
        ctx.save_for_backward(ids)
        ctx.weight_ref = weight
        return weight[ids]

    @staticmethod
    def backward(ctx, grad_output):
        ids = ctx.saved_tensor()[0]
        weight = ctx.weight_ref
        # E-530 dump-only: live dY of each IndexingBackward (fused vs two).
        # Observation; returning grad_output unchanged. Gated so dump-off
        # 10-step bits stay E-527 when the env is unset.
        # Hit counter is monotonic: the first draft reused _printed and
        # decremented it, so every step overwrote h1/h2.
        dump_dy = os.environ.get("MODEL_REPRO_EMBED_DY_DIR")
        hashdir = os.environ.get("MODEL_REPRO_EMBED_DY_HASH_DIR")
        if dump_dy or hashdir:
            import hashlib
            import json

            _EmbedFp32MainGrad._dy_hits += 1
            hit = _EmbedFp32MainGrad._dy_hits
            try:
                rank = int(dist.get_rank()) if dist.is_initialized() else 0
            except Exception:
                rank = 0
            dy = grad_output.detach().astype("float32").numpy()
            idn = ids.detach().numpy()
            meta = {
                "kind": "bwd",
                "tag": "embed_dy",
                "framework": "paddle",
                "rank": rank,
                "hit": hit,
                "ids_shape": list(idn.shape),
                "dy_shape": list(dy.shape),
                "ids_n": int(idn.size),
                "dy_sha256": hashlib.sha256(dy.tobytes()).hexdigest(),
                "ids_sha256": hashlib.sha256(idn.tobytes()).hexdigest(),
            }
            if hashdir:
                os.makedirs(hashdir, exist_ok=True)
                with open(
                    os.path.join(hashdir, f"rank{rank}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
                if hit == 1:
                    print(
                        f"[E546-EMBED-DY-HASH] dir={hashdir} rank={rank} hit={hit} "
                        f"ids_n={meta['ids_n']} sha={meta['dy_sha256'][:16]}",
                        flush=True,
                    )
            if dump_dy:
                os.makedirs(dump_dy, exist_ok=True)
                stem = (
                    f"paddle_embed_dy_r{rank}_h{hit}_L{int(idn.size)}"
                )
                dy.tofile(os.path.join(dump_dy, f"{stem}.f32.bin"))
                idn.tofile(os.path.join(dump_dy, f"{stem}_ids.i64.bin"))
                with open(
                    os.path.join(dump_dy, f"{stem}.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(meta, handle, sort_keys=True)
                    handle.write("\n")
                print(
                    f"[EMBED-DY-DUMP] r{rank} hit={hit} "
                    f"ids={tuple(idn.shape)} dy={tuple(dy.shape)} "
                    f"sha={meta['dy_sha256'][:16]}",
                    flush=True,
                )
        # Function.backward disables grads (E-472 gw=None). Re-enable so
        # W[ids] is IndexingBackward on a clone, not embedding_grad_add_to_
        # (E-468 nz 233424) and not MixPrecision bf16 merge (E-470 hit=1).
        # iso_e473: two hits, each gw.cast(fp32), add == native two-fp32.
        prev = paddle.is_grad_enabled()
        paddle.set_grad_enabled(True)
        try:
            w = weight.detach().clone()
            w.stop_gradient = False
            looked = w[ids]
            (gw,) = paddle.autograd.grad(
                looked, w, grad_output, allow_unused=True
            )
        finally:
            paddle.set_grad_enabled(prev)
        _EmbedFp32MainGrad._printed += 1
        if gw is None:
            print(
                "[TWO-FP32-ACCUM] pylayer gw=None "
                f"hit={_EmbedFp32MainGrad._printed} "
                f"ids={tuple(ids.shape)}",
                flush=True,
            )
            return None, None
        fp = gw.cast(paddle.float32)
        if hasattr(weight, "main_grad") and weight.main_grad is not None:
            weight.main_grad.add_(fp)
        else:
            weight.main_grad = fp
        if hasattr(weight, "grad_added_to_main_grad"):
            weight.grad_added_to_main_grad = True
        if _EmbedFp32MainGrad._printed <= 4:
            print(
                "[TWO-FP32-ACCUM] pylayer IndexingBackward "
                f"hit={_EmbedFp32MainGrad._printed} "
                f"ids={tuple(ids.shape)} gw_dtype={gw.dtype} "
                f"gw_nz={(gw != 0).astype('int64').sum().item()}",
                flush=True,
            )
        return None, None


class VocabParallelEmbedding(paddle.nn.Layer):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from paddle.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup

    Keyword Args:
        config: A fleet.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: callable,
        reduce_scatter_embeddings: bool = False,
        config: TransformerConfig,
        tp_group: paddle.distributed.ProcessGroup | None = None,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tp_group = tp_group
        self._dtype = config.params_dtype

        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, check_initialized=False
        )

        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_pg_rank(self.tp_group),
                get_pg_size(self.tp_group),
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.deterministic_mode = config.deterministic_mode
        self.use_accuracy_compatible = getattr(
            config, "use_accuracy_compatible", False
        )
        self.config = config
        self.world_size = get_pg_size(self.tp_group)

        # Allocate weights and initialize.
        if config.use_cpu_initialization:
            self.weight = self.create_parameter(
                shape=[self.num_embeddings_per_partition, self.embedding_dim],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=get_pg_rank(self.tp_group),
                    world_size=get_pg_size(self.tp_group),
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.num_embeddings_per_partition, self.embedding_dim],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=0, stride=1
                )
        self.weight.is_distributed = True if self.world_size > 1 else False

    def forward(self, input_):
        """Forward.

        Args:
            input_ (paddle.Tensor): Input tensor.
        """
        if getattr(self.config, "gpt_model_use_experimental_version", False):
            return vocab_parallel_embedding(
                input_,
                self.weight,
                vocab_start_index=self.vocab_start_index,
                num_embeddings=self.num_embeddings,
                mp_group=self.tp_group,
            )

        if get_pg_size(self.tp_group) > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            input_mask = None
        _dump_embed_lookup_ids(
            input_,
            masked_input,
            input_mask,
            self.vocab_start_index,
            self.vocab_end_index,
        )
        # E-576 dump-off: lookup W hash once per rank. Observation only.
        dump_w = os.environ.get("MODEL_REPRO_EMBED_W_HASH_DIR")
        if dump_w and not getattr(self, "_e576_w_hashed", False):
            import hashlib
            import json

            try:
                rank = int(dist.get_rank()) if dist.is_initialized() else 0
            except Exception:
                rank = 0
            os.makedirs(dump_w, exist_ok=True)
            w = self.weight.detach().cpu()
            rec = {
                "kind": "param",
                "tag": "embed_w",
                "framework": "paddle",
                "rank": int(rank),
                "shape": list(w.shape),
                "dtype": str(w.dtype),
                "sha_w": hashlib.sha256(
                    w.contiguous().view(dtype="uint16").numpy().tobytes()
                    if "bfloat16" in str(w.dtype)
                    else w.contiguous().numpy().tobytes()
                ).hexdigest(),
                "vocab_start": int(self.vocab_start_index),
                "vocab_end": int(self.vocab_end_index),
            }
            with open(os.path.join(dump_w, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[E576-EMBED-W-HASH] dir={dump_w} rank={rank} shape={rec['shape']} "
                f"sha={rec['sha_w'][:16]}",
                flush=True,
            )
            self._e576_w_hashed = True
        # E-578 dump-off: W at fused-169 / MTP-168 calls, not first lookup.
        dump_w_fused = os.environ.get("MODEL_REPRO_EMBED_W_AT_FUSED_HASH_DIR")
        if dump_w_fused:
            import hashlib
            import json

            idn = int(masked_input.numel())
            if idn in (168, 169):
                try:
                    rank = int(dist.get_rank()) if dist.is_initialized() else 0
                except Exception:
                    rank = 0
                if not hasattr(_EmbedFp32MainGrad, "_e578_w_hits"):
                    _EmbedFp32MainGrad._e578_w_hits = 0
                _EmbedFp32MainGrad._e578_w_hits += 1
                hit = _EmbedFp32MainGrad._e578_w_hits
                os.makedirs(dump_w_fused, exist_ok=True)
                w = self.weight.detach().cpu()
                rec = {
                    "kind": "param",
                    "tag": "embed_w_at_fused",
                    "framework": "paddle",
                    "rank": int(rank),
                    "hit": int(hit),
                    "ids_n": int(idn),
                    "shape": list(w.shape),
                    "dtype": str(w.dtype),
                    "sha_w": hashlib.sha256(
                        w.contiguous().view(dtype="uint16").numpy().tobytes()
                        if "bfloat16" in str(w.dtype)
                        else w.contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "vocab_start": int(self.vocab_start_index),
                    "vocab_end": int(self.vocab_end_index),
                }
                with open(
                    os.path.join(dump_w_fused, f"rank{rank}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[E578-EMBED-W-AT-FUSED-HASH] dir={dump_w_fused} rank={rank} "
                    f"hit={hit} ids_n={idn} sha={rec['sha_w'][:16]}",
                    flush=True,
                )
        # E-579 dump-off: unique W rows at fused-169, not the 905MiB table.
        dump_rows = os.environ.get("MODEL_REPRO_EMBED_W_ROWS_DIR")
        if dump_rows and int(masked_input.numel()) == 169:
            import hashlib
            import json

            try:
                rank = int(dist.get_rank()) if dist.is_initialized() else 0
            except Exception:
                rank = 0
            if not getattr(self, "_e579_rows_dumped", False):
                os.makedirs(dump_rows, exist_ok=True)
                ids_full = masked_input.detach().reshape([-1])
                ids_pref = ids_full[:-1]
                uniq = paddle.unique(ids_pref)
                rows = self.weight.detach()[uniq].contiguous().cpu()
                uniq_np = uniq.cpu().numpy()
                rows_u16 = (
                    rows.view(dtype="uint16").numpy()
                    if "bfloat16" in str(rows.dtype)
                    else rows.numpy()
                )
                stem = f"paddle_embed_w_rows_r{rank}_n169"
                uniq_np.astype("int64").tofile(os.path.join(dump_rows, f"{stem}_ids.i64.bin"))
                rows_u16.tofile(os.path.join(dump_rows, f"{stem}.u16.bin"))
                rec = {
                    "kind": "param",
                    "tag": "embed_w_rows",
                    "framework": "paddle",
                    "rank": int(rank),
                    "ids_n": 169,
                    "prefix_n": 168,
                    "n_unique": int(uniq_np.size),
                    "shape_rows": list(rows.shape),
                    "sha_rows": hashlib.sha256(
                        rows_u16.tobytes()
                    ).hexdigest(),
                    "sha_uniq_ids": hashlib.sha256(uniq_np.tobytes()).hexdigest(),
                    "vocab_start": int(self.vocab_start_index),
                    "vocab_end": int(self.vocab_end_index),
                    "sha_w_full": hashlib.sha256(
                        self.weight.detach()
                        .contiguous()
                        .cpu()
                        .view(dtype="uint16")
                        .numpy()
                        .tobytes()
                    ).hexdigest()
                    if "bfloat16" in str(self.weight.dtype)
                    else None,
                }
                with open(
                    os.path.join(dump_rows, f"{stem}.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(rec, handle, sort_keys=True)
                with open(
                    os.path.join(dump_rows, f"rank{rank}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[E579-EMBED-W-ROWS] dir={dump_rows} rank={rank} "
                    f"n_unique={rec['n_unique']} sha_w={rec['sha_w_full'][:16] if rec['sha_w_full'] else None}",
                    flush=True,
                )
                self._e579_rows_dumped = True
        # Get the embeddings.
        if self.deterministic_mode or self.use_accuracy_compatible:
            if os.environ.get("MODEL_REPRO_TWO_FP32_ACCUM", "") == "1":
                output_parallel = _EmbedFp32MainGrad.apply(
                    self.weight, masked_input
                )
            else:
                output_parallel = self.weight[masked_input]
        else:
            # F.embedding currently has a non-deterministic backward function
            output_parallel = F.embedding(masked_input, self.weight)
        # E-574 dump-off: fused lookup Y (pre-OOV-mask) + prefix dY[:-1]
        # vs torch first n=S-1. Separate env from EMBED_DY / SLICE / FSLN.
        dump_y = os.environ.get("MODEL_REPRO_EMBED_Y_HASH_DIR")
        if dump_y:
            import hashlib
            import json

            try:
                rank = int(dist.get_rank()) if dist.is_initialized() else 0
            except Exception:
                rank = 0
            if not hasattr(_EmbedFp32MainGrad, "_y_hits"):
                _EmbedFp32MainGrad._y_hits = 0
            _EmbedFp32MainGrad._y_hits += 1
            hit = _EmbedFp32MainGrad._y_hits
            os.makedirs(dump_y, exist_ok=True)
            y = output_parallel.detach().cpu()
            idn = masked_input.detach()
            seq = int(y.shape[1]) if y.ndim >= 2 else int(y.shape[0])
            y_prefix = y[:, :-1] if y.ndim >= 2 and seq > 1 else y
            rec = {
                "kind": "fwd",
                "tag": "embed_y",
                "framework": "paddle",
                "rank": int(rank),
                "hit": int(hit),
                "ids_n": int(idn.size),
                "shape_y": list(y.shape),
                "sha_y": hashlib.sha256(
                    y.contiguous().view(dtype="uint16").numpy().tobytes()
                    if "bfloat16" in str(y.dtype)
                    else y.contiguous().numpy().tobytes()
                ).hexdigest(),
                "shape_y_prefix": list(y_prefix.shape),
                "sha_y_prefix": hashlib.sha256(
                    y_prefix.contiguous().view(dtype="uint16").numpy().tobytes()
                    if "bfloat16" in str(y_prefix.dtype)
                    else y_prefix.contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            with open(os.path.join(dump_y, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if hit == 1:
                print(
                    f"[E575-EMBED-Y-HASH] dir={dump_y} rank={rank} hit={hit} "
                    f"ids_n={rec['ids_n']}",
                    flush=True,
                )

            def _on_embed_y_dy(g, *, _dump=dump_y, _rank=rank, _hit=hit, _ids_n=int(idn.size)):
                if g is None:
                    return g
                g_cpu = g.detach().cpu()
                seq = int(g_cpu.shape[1]) if g_cpu.ndim >= 2 else int(g_cpu.shape[0])
                prefix = g_cpu[:, :-1] if g_cpu.ndim >= 2 and seq > 1 else g_cpu
                extra = g_cpu[:, -1:] if g_cpu.ndim >= 2 and seq > 1 else None
                bwd = {
                    "kind": "bwd",
                    "tag": "embed_y",
                    "framework": "paddle",
                    "rank": int(_rank),
                    "hit": int(_hit),
                    "ids_n": int(_ids_n),
                    "shape_dy": list(g_cpu.shape),
                    "sha_dy": hashlib.sha256(
                        g_cpu.contiguous().view(dtype="uint16").numpy().tobytes()
                        if "bfloat16" in str(g_cpu.dtype)
                        else g_cpu.contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "shape_dy_prefix": list(prefix.shape),
                    "sha_dy_prefix": hashlib.sha256(
                        prefix.contiguous().view(dtype="uint16").numpy().tobytes()
                        if "bfloat16" in str(prefix.dtype)
                        else prefix.contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
                if extra is not None:
                    bwd["shape_dy_extra"] = list(extra.shape)
                    bwd["sha_dy_extra"] = hashlib.sha256(
                        extra.contiguous().view(dtype="uint16").numpy().tobytes()
                        if "bfloat16" in str(extra.dtype)
                        else extra.contiguous().numpy().tobytes()
                    ).hexdigest()
                with open(os.path.join(_dump, f"rank{_rank}.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(bwd, ensure_ascii=False) + "\n")
                return g

            if getattr(output_parallel, "stop_gradient", True) is False:
                output_parallel.register_hook(_on_embed_y_dy)
        # Mask the output embedding.
        if get_pg_size(self.tp_group) > 1:
            output_parallel[input_mask, :] = 0.0

        if self.reduce_scatter_embeddings:
            # Data format change to avoid explicit transpose : [b s h] --> [s b h].
            # output_parallel = output_parallel.transpose(0, 1).contiguous()
            output_parallel = output_parallel.transpose([1, 0, 2]).contiguous()
            output = reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            # Reduce across all the model parallel GPUs.
            output = reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        output = _maybe_dump_embed_chain(output, "allreduce")
        return output

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )


class LinearWithFrozenWeight(paddle.autograd.Function):
    """Linear operator that does not calculate gradient for weight.
    This op and LinearWithGradAccumulationAndAsyncCommunication performs
    mathematically-identical forward and DGRAD.

    Conceptually this op is the same as linear with weight.requires_grad==False,
    but in experiments they are not identical mathematically."""

    @staticmethod
    def forward(ctx, input, weight, bias, allreduce_dgrad, tp_group):
        """Forward with frozen weight."""
        ctx.save_for_backward(weight, bias)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        output = paddle.matmul(input, weight)

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward with frozen weight."""
        (weight, bias) = ctx.saved_tensor()
        grad_input = grad_output.matmul(weight.t())

        if ctx.allreduce_dgrad:
            # All-reduce. Note: here async and sync are effectively the same.
            dist.all_reduce(grad_input, group=ctx.tp_group)
        if bias is None:
            return grad_input, None
        else:
            return grad_input, None, None


def linear_with_frozen_weight(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    tp_group: paddle.core.ProcessGroup | None,
    grad_output_buffer: list[paddle.Tensor] | None = None,
    wgrad_deferral_limit: None = None,
    async_grad_allreduce: bool | None = None,
    use_accuracy_compatible: bool = False,
    **kwargs,
) -> paddle.Tensor:
    """Linear layer execution with weight.requires_grad == False.

    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.

    Args:

    input (paddle.Tensor required): input like paddle.nn.functional.linear

    weight (paddle.Tensor required): weight like paddle.nn.functional.linear

    bias (paddle.Tensor optional): bias like paddle.nn.functional.linear

    gradient_accumulation_fusion (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    allreduce_dgrad (bool, required): Do the allreduce of input gradients.
        Here, async and sync allreduce are the same. If sequence_parallel is
        True, this must be False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.

    tp_group (paddle.core.ProcessGroup): The process group to use for tensor
                                                       parallel operations.

    grad_output_buffer (List[paddle.Tensor] optional): dummy argument, used to
    keep the API unified between all forward implementation functions.

    wgrad_deferral_limit (int optional): dummy argument, used to
    keep the API unified between all forward implementation functions.


    async_grad_allreduce (bool optional): Will be removed with 0.11.0.
                                          Please use allreduce_dgrad instead.

    """

    if async_grad_allreduce is not None:
        warnings.warn(
            "async_grad_allreduce is deprecated, not in use anymore and will"
            " be fully removed with 0.11.0. Please use allreduce_dgrad instead."
        )

    assert grad_output_buffer is None, (
        "grad_output_buffer kwarg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    assert wgrad_deferral_limit is None, (
        "This arg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    if sequence_parallel:
        input = gather_from_sequence_parallel_region(
            input, tensor_parallel_output_grad=True, group=tp_group
        )
    else:
        input = input

    args = [input, weight, bias, allreduce_dgrad, tp_group]

    return LinearWithFrozenWeight.apply(*args)


def _bwd_quant_blockwise_1x128(x, use_pow2_scale, use_ue8m0):
    """Backward-path 1x128 blockwise quant.

    When ``use_ue8m0=True`` we produce int32-packed pow2 scales in MN-major
    layout (``output_scale_transpose=True`` + ``.T`` stride-only view) so
    DeepGEMM's SM100 INT/(1,1,128) branch can consume them without extra
    H2D transfers. When ``use_ue8m0=False`` we keep the original behavior
    (fp32 scales, ``output_scale_transpose=False``) which matches the
    pre-existing pow2-only golden.
    """
    if use_ue8m0:
        fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_pow2_scale=True,
            using_ue8m0_scale=True,
        )[:2]
        return fp8, scale.T
    return paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        output_scale_transpose=False,
        quant_method="1x128",
        input_transpose=False,
        using_pow2_scale=use_pow2_scale,
    )[:2]


def _dequant_inp_t_to_bf16(inp_t_fp8, inp_t_scale):
    """Dequantize the pre-quantized ``[K, M]`` fp8 activation back to bf16.

    Used by the backward wgrad fallback when ``save_original_input=False``
    (so the bf16 activation was not saved) and ``fp8_wgrad=False`` (so the
    wgrad gemm still runs in bf16). ``inp_t_fp8`` is the transposed-
    orientation fp8 quantization emitted by the forward quant kernel,
    with shape ``[K, M]`` — exactly what ``total_input.t()`` would be.

    ``inp_t_scale`` was passed through ``_mn_major`` at forward time
    (stride-only ``.T`` view for DeepGEMM's ``stride(-2)==1`` requirement).
    ``fused_act_dequant`` requires K-major scale (``stride(-1)==1``), so
    we materialize a contiguous copy here. Handles both float32 scales
    and UE8M0 int32-packed scales transparently.
    """
    scale_k_major = inp_t_scale.contiguous()
    return paddle.incubate.nn.functional.fused_act_dequant(
        inp_t_fp8, scale_k_major
    )


def _make_bwd_inp_quant_func(use_pow2_scale, use_ue8m0):
    """Return an ``inp_quant_func`` callable for the backward dgrad path.

    Mirrors ``_bwd_quant_blockwise_1x128`` but returns a callable in the
    shape expected by ``general_gemm``'s ``inp_quant_func`` argument.
    """

    def _f(x):
        return _bwd_quant_blockwise_1x128(x, use_pow2_scale, use_ue8m0)

    return _f


def general_gemm(
    a,
    b,
    bias=None,
    fp8=False,
    inp_quant_func=None,
    weight_quant_func=None,
    out=None,
    recipe=None,
    use_accuracy_compatible=False,
):
    """Unified GEMM interface supporting both bf16 and fp8 paths.

    Args:
        a: Left operand. Raw tensor or pre-quantized (fp8_tensor, scale) tuple.
        b: Right operand. Same convention as *a*.
        bias: Optional bias (only used in bf16 path).
        fp8: If True, use fp8 gemm via deep_gemm.fp8_gemm_nt.
        inp_quant_func: Quantization callable for input (when fp8=True).
        weight_quant_func: Quantization callable for weight (when fp8=True).
        out: Pre-allocated output buffer. If not None, accumulates into it;
             if None, creates a new buffer.

    Returns:
        output tensor, and optionally quant cache tuple when fp8=True.
    """
    if fp8:
        assert _deep_gemm_available, (
            "FP8 GEMM requires paddlefleet_ops.deep_gemm"
        )

        from paddlefleet.fp8.utils import is_fp8_tensor

        # fp8_quant_blockwise requires 2D input, reshape if needed
        a_orig_shape = None
        if not is_fp8_tensor(a) and a.ndim > 2:
            a_orig_shape = a.shape
            a = a.reshape([-1, a.shape[-1]])

        if is_fp8_tensor(a):
            inp_fp8, inp_scale = a
            inp_t_fp8, inp_t_scale = None, None
        else:
            quant_result = inp_quant_func(a)
            if len(quant_result) == 2:
                inp_fp8, inp_scale = quant_result
                inp_t_fp8, inp_t_scale = None, None
            else:
                inp_fp8, inp_scale, inp_t_fp8, inp_t_scale = quant_result

        # weight_quant_func returns either a 2-tuple ``(fp8, scale)`` or a
        # 4-tuple ``(fp8_bwd, scale_bwd, fp8_fwd, scale_fwd)``. The forward
        # orientation feeds fp8_gemm_nt here; the backward orientation (if
        # present) is returned for dgrad reuse.
        weight_fp8_bwd, weight_scale_bwd = None, None
        if is_fp8_tensor(b):
            weight_fp8, weight_scale = b
        else:
            wq_result = weight_quant_func(b)
            if len(wq_result) == 2:
                weight_fp8, weight_scale = wq_result
            else:
                (
                    weight_fp8_bwd,
                    weight_scale_bwd,
                    weight_fp8,
                    weight_scale,
                ) = wq_result

        if out is not None:
            # Accumulate into existing buffer
            deep_gemm.fp8_gemm_nt(
                (inp_fp8, inp_scale),
                (weight_fp8, weight_scale),
                out,
                c=out,
                recipe=recipe,
            )
        else:
            out = paddle.empty(
                [inp_fp8.shape[0], weight_fp8.shape[0]], dtype=paddle.bfloat16
            )
            deep_gemm.fp8_gemm_nt(
                (inp_fp8, inp_scale),
                (weight_fp8, weight_scale),
                out,
                recipe=recipe,
            )

        if a_orig_shape is not None:
            out = out.reshape([*list(a_orig_shape[:-1]), out.shape[-1]])

        return out, (
            inp_fp8,
            inp_scale,
            inp_t_fp8,
            inp_t_scale,
            weight_fp8,
            weight_scale,
            weight_fp8_bwd,
            weight_scale_bwd,
        )

    else:
        # Standard bf16/fp16 path
        if bias is not None:
            output = paddle.nn.functional.linear(a, b, bias)
        else:
            if os.environ.get("MODEL_REPRO_MOE_TN_GEMM", "0") == "1":
                # E-107: mcore runs torch.matmul(x, w.t()) with weight stored
                # [out, in] contiguous (a TN GEMM); PaddleFleet stores [in, out]
                # and issues an NN GEMM. cuBLAS selects different kernels and they
                # differ bitwise at exactly the (M,K,N) layer-3 uses (routed fc1
                # K6144->N4096 at ragged M; shared fc1 K6144->N2048 at M=60) while
                # agreeing at the dense M=60 shapes. Materializing [out, in] is
                # required: a transposed view shares storage and still dispatches NN.
                output = paddle.matmul(a, b.t().contiguous(), transpose_y=True)
            elif use_accuracy_compatible:
                output = paddle.nn.functional.linear(a, b)
            else:
                output = paddle.matmul(a, b)
        return output, None


class LinearWithGradAccumulationAndAsyncCommunication(paddle.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        use_accuracy_compatible=False,
        fp8=False,
        fp8_wgrad=False,
        inp_quant_func=None,
        weight_quant_func=None,
        use_pow2_scale=False,
        use_ue8m0=False,
        save_original_input=False,
    ):
        """Forward."""
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        # When fp8 is on we can skip the bf16 activation and use the fp8
        # cache on ctx; ``save_original_input`` forces the bf16 tensor to
        # be kept (needed for bf16 wgrad).
        skip_bf16_input_save = fp8 and not save_original_input
        ctx.bf16_input_saved = not skip_bf16_input_save
        # ``input.shape`` is still needed in backward for sequence-parallel
        # collectives; save as a plain tuple to avoid retaining the tensor.
        ctx.input_shape = tuple(input.shape)
        ctx.input_dtype = input.dtype
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.tp_group = tp_group
        ctx.use_accuracy_compatible = use_accuracy_compatible
        # Cache input.stop_gradient: ``_new_shared_tensor()`` does not
        # necessarily preserve this flag, and Paddle's PyLayer contract
        # requires backward to return None at position 0 iff the original
        # forward input had stop_gradient=True.
        ctx.input_stop_gradient = bool(input.stop_gradient)
        ctx.fp8 = fp8
        ctx.fp8_wgrad = fp8_wgrad
        ctx.inp_quant_func = inp_quant_func
        ctx.weight_quant_func = weight_quant_func
        ctx.use_pow2_scale = use_pow2_scale
        ctx.use_ue8m0 = use_ue8m0
        ctx.save_original_input = save_original_input

        if sequence_parallel:
            dim_size = list(input.shape)
            dim_size[0] = dim_size[0] * tp_group.world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            dist.stream.all_gather(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output, fp8_meta = general_gemm(
            total_input,
            weight,
            bias=bias,
            fp8=fp8,
            inp_quant_func=inp_quant_func,
            weight_quant_func=weight_quant_func,
            use_accuracy_compatible=use_accuracy_compatible,
        )

        save_list = []
        if not skip_bf16_input_save:
            save_list.append(input._new_shared_tensor())
        save_list.append(weight)
        if fp8 and fp8_meta is not None:
            (
                _,
                _,
                inp_t_fp8,
                inp_t_scale,
                weight_fp8,
                weight_scale,
                weight_fp8_bwd,
                weight_scale_bwd,
            ) = fp8_meta
            # When ``save_original_input`` is True the bf16 activation is kept
            # and backward re-quantizes it, so ``inp_t_fp8``/``inp_t_scale``
            # can be dropped.
            if save_original_input:
                save_list.extend([weight_fp8, weight_scale, weight_scale_bwd])
                ctx.fp8_input_stashed = False
            else:
                save_list.extend(
                    [
                        inp_t_fp8,
                        inp_t_scale,
                        weight_fp8,
                        weight_scale,
                        weight_scale_bwd,
                    ]
                )
                ctx.fp8_input_stashed = True
            ctx.fp8_saved = True
        else:
            ctx.fp8_saved = False
            ctx.fp8_input_stashed = False
        ctx.save_for_backward(*save_list)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward."""
        saved = ctx.saved_tensor()
        idx = 0
        if ctx.bf16_input_saved:
            input = saved[idx]
            idx += 1
        else:
            input = None
        weight = saved[idx]
        idx += 1
        if ctx.fp8_saved:
            if ctx.fp8_input_stashed:
                inp_t_fp8 = saved[idx]
                idx += 1
                inp_t_scale = saved[idx]
                idx += 1
            else:
                inp_t_fp8 = None
                inp_t_scale = None
            weight_fp8 = saved[idx]
            idx += 1
            weight_scale = saved[idx]
            idx += 1
            weight_scale_bwd = saved[idx]
            idx += 1
        else:
            inp_t_fp8 = None
            inp_t_scale = None
            weight_fp8 = None
            weight_scale = None
            weight_scale_bwd = None
        # weight_fp8_bwd is reconstructed below via weight_fp8.T.contiguous().
        weight_fp8_bwd = None
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        handle = None
        tp_group = ctx.tp_group
        fp8 = ctx.fp8
        fp8_wgrad = ctx.fp8_wgrad

        input_needs_grad = not ctx.input_stop_gradient

        # AMP casts the activation to the weight dtype inside the forward gemm,
        # but this PyLayer differentiates the gemms by hand and the backward
        # runs with AMP disabled (activation recompute explicitly turns it off),
        # so the cast has to be reproduced here. Otherwise a fp32 activation
        # feeding a bf16 weight (e.g. the fp32 embedding output produced by
        # ``fp32_residual_connection``) makes the wgrad gemm fail on mixed
        # dtypes. The dgrad is cast back to the original input dtype below.
        if input is not None and input.dtype != weight.dtype:
            input = input.astype(weight.dtype)

        if ctx.gradient_accumulation_fusion:
            weight.main_grad = main_grad

        wgrad_compute = True
        if grad_output_buffer is not None:
            if (
                wgrad_deferral_limit == 0
                or len(grad_output_buffer) < wgrad_deferral_limit
            ):
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                dim_size = list(input.shape)
                dim_size[0] = dim_size[0] * tp_group.world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = dist.stream.all_gather(
                    all_gather_buffer, input, group=tp_group, sync_op=False
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input

        # Compute grad_input
        if input_needs_grad:
            if fp8 and weight_fp8 is not None:
                # FP8 dgrad: grad_input = grad_output @ weight
                # Reuse the backward orientation quantized in forward when
                # available; otherwise fall back to `.T.contiguous()`.
                if weight_fp8_bwd is not None:
                    weight_bwd = (weight_fp8_bwd, weight_scale_bwd)
                else:
                    weight_bwd = (
                        weight_fp8.T.contiguous(),
                        weight_scale_bwd,
                    )
                grad_input, _ = general_gemm(
                    grad_output,
                    weight_bwd,
                    fp8=True,
                    inp_quant_func=_make_bwd_inp_quant_func(
                        ctx.use_pow2_scale, ctx.use_ue8m0
                    ),
                )
            else:
                if ctx.use_accuracy_compatible:
                    # E-240: THE DGRAD OPERAND LAYOUT. ``weight`` is stored
                    # [in, out] here, so ``weight.t()`` is a transposed VIEW that
                    # shares storage and makes cuBLAS pick its transposed-B
                    # kernel. The reference stores the weight [out, in]
                    # contiguous and computes ``grad_output @ weight``, an NN
                    # GEMM. Those two kernels reduce the K = 6144 dimension in a
                    # different order and disagree bitwise.
                    #
                    # E-239 measured the consequence: this was the FIRST
                    # divergence left after the E-235 token-normalization fix. At
                    # the MTP layer's shared-expert down projection the forward is
                    # bit-equal at every internal boundary and the incoming
                    # gradient is bit-equal (rank3 184,320/184,320, abssum deficit
                    # exactly 0), yet this dgrad differed in 170 of 61,440
                    # elements - 154 of them by exactly 1 ulp, symmetric in
                    # direction (85 smaller / 85 larger), and concentrated on the
                    # SMALLEST elements (median magnitude 1.29e-04 against
                    # 3.68e-03 overall, the 2.4th percentile). That is
                    # cancellation in a long reduction summed in a different
                    # order, not a wrong factor.
                    #
                    # E-244 settled which half of that story is load-bearing, by
                    # replaying the operands dumped from INSIDE both backward
                    # functions (M=60, K=6144, N=1024, bf16):
                    #   * with the [out, in] operand MATERIALIZED, both
                    #     frameworks reproduce the reference's in-function dgrad
                    #     bit-for-bit (0 of 61,440 differing, rank2 and rank3);
                    #   * left as a transposed VIEW, BOTH frameworks miss the
                    #     reference by the SAME 116 (rank2) / 107 (rank3)
                    #     elements.
                    # Materialization is therefore necessary and sufficient here,
                    # and the view difference is a property of the GEMM operand
                    # layout rather than a cross-framework defect - the same
                    # conclusion E-107 reached for the FORWARD GEMM (see the
                    # MODEL_REPRO_MOE_TN_GEMM branch in general_gemm above).
                    #
                    # The 2-D flatten below matches the reference's dispatch
                    # shape (Megatron's local linear path flattens the leading
                    # dimensions before torch.matmul, while grad_output here is
                    # commonly [M, 1, K]). E-244 measured it as numerically INERT
                    # at every shape in this profile: paddle's batched
                    # [M,1,K] @ [K,N] and its flattened form agree in
                    # 61,440/61,440, 122,880/122,880, 368,640/368,640 and
                    # 491,520/491,520 elements. It is kept because it makes the
                    # dispatch shape match the reference by construction, NOT
                    # because it changes any bits. Default numerics unchanged.
                    original_shape = grad_output.shape
                    flat_grad_output = grad_output.reshape([-1, original_shape[-1]])
                    flat_grad_input = paddle.matmul(
                        flat_grad_output, weight.t().contiguous()
                    )
                    grad_input = flat_grad_input.reshape(
                        list(original_shape[:-1]) + [weight.shape[0]]
                    )
                else:
                    grad_input, _ = general_gemm(grad_output, weight.t())
        else:
            grad_input = None

        if ctx.sequence_parallel and wgrad_compute:
            # pylint: disable=possibly-used-before-assignment
            handle.wait()

        if wgrad_compute:
            if total_input is not None:
                grad_output, total_input = (
                    prepare_input_tensors_for_wgrad_compute(
                        grad_output, total_input
                    )
                )
            else:
                # fp8 + save_original_input=False: no bf16 input available.
                # Only reshape grad_output to 2D for the fp8 wgrad path.
                grad_output = grad_output.contiguous()
                if grad_output.dim() == 3:
                    grad_output = grad_output.reshape(
                        [
                            grad_output.shape[0] * grad_output.shape[1],
                            grad_output.shape[2],
                        ]
                    )

        if ctx.allreduce_dgrad and input_needs_grad:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=tp_group, sync_op=False)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            if input_needs_grad:
                dim_size = list(input.shape)
                sub_grad_input = paddle.empty(
                    dim_size, dtype=input.dtype, requires_grad=False
                )
                # reduce_scatter
                handle = _reduce_scatter_base(
                    sub_grad_input, grad_input, group=tp_group, sync_op=False
                )
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # reduce scatter is scheduled before the weight gradient computation
            else:
                sub_grad_input = None

        # Compute grad_weight
        if fp8_wgrad and wgrad_compute:
            # FP8 wgrad: grad_output^T @ total_input (both already 2D)
            grad_out_t_fp8, grad_out_t_scale = _bwd_quant_blockwise_1x128(
                grad_output.T.contiguous(),
                ctx.use_pow2_scale,
                ctx.use_ue8m0,
            )
            inp_t_fp8_bwd, inp_t_scale_bwd = inp_t_fp8, inp_t_scale
            if inp_t_fp8_bwd is None:
                assert total_input is not None, (
                    "fp8 wgrad requires either pre-quantized inp_t_fp8 (from"
                    " fp8 forward with input_trans=True) or the bf16 total_input"
                    " (set save_original_input=True to keep bf16 activation)"
                )
                inp_t_fp8_bwd, inp_t_scale_bwd = _bwd_quant_blockwise_1x128(
                    total_input.T.contiguous(),
                    ctx.use_pow2_scale,
                    ctx.use_ue8m0,
                )

            grad_weight, _ = general_gemm(
                (inp_t_fp8_bwd, inp_t_scale_bwd),
                (grad_out_t_fp8, grad_out_t_scale),
                fp8=True,
                recipe=(1, 1, 128),
            )

        elif ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == paddle.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (
                    paddle.float16,
                    paddle.bfloat16,
                ):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError(
                        "Unsupported gradient type for gradient accumulation fusion"
                    )

            if hasattr(weight, "grad_added_to_main_grad"):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, "zero_out_wgrad", False):
                    grad_weight = paddle.zeros(
                        weight.main_grad.shape,
                        dtype=ctx.input_dtype,
                        requires_grad=False,
                    )
                else:
                    grad_weight = paddle.empty(
                        weight.main_grad.shape,
                        dtype=ctx.input_dtype,
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            if (
                wgrad_compute
                and ctx.use_accuracy_compatible
                and getattr(weight, "is_expert_param", False)
                and hasattr(weight, "main_grad")
                and weight.main_grad is not None
                and weight.main_grad.dtype == paddle.float32
            ):
                weight.main_grad.add_(
                    paddle.matmul(
                        grad_output.astype("float32"),
                        total_input.astype("float32"),
                        transpose_x=True,
                    ).t()
                )
                if hasattr(weight, "grad_added_to_main_grad"):
                    weight.grad_added_to_main_grad = True
                grad_weight = paddle.zeros(weight.shape, dtype=input.dtype)
            elif wgrad_compute:
                if total_input is not None:
                    grad_weight, _ = general_gemm(total_input.t(), grad_output)
                elif inp_t_fp8 is not None:
                    # No bf16 input saved; dequantize the fp8 transposed
                    # activation (shape [K, M] = total_input.t()) for bf16 wgrad.
                    total_input_t_bf16 = _dequant_inp_t_to_bf16(
                        inp_t_fp8, inp_t_scale
                    )
                    grad_weight, _ = general_gemm(
                        total_input_t_bf16, grad_output
                    )
                else:
                    raise AssertionError(
                        "bf16 wgrad requires either the saved bf16 "
                        "activation (save_original_input=True) or a "
                        "pre-quantized fp8 activation saved for backward "
                        "(fp8 forward with input_trans=True)"
                    )
            else:
                grad_weight = None
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            if input_needs_grad:
                handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            if sub_grad_input is not None and (
                sub_grad_input.dtype != ctx.input_dtype
            ):
                sub_grad_input = sub_grad_input.astype(ctx.input_dtype)
            if use_bias:
                return sub_grad_input, grad_weight, grad_bias
            else:
                return sub_grad_input, grad_weight

        if ctx.allreduce_dgrad and input_needs_grad:
            handle.wait()

        if grad_input is not None and grad_input.dtype != ctx.input_dtype:
            grad_input = grad_input.astype(ctx.input_dtype)

        # PyLayer requires the number of output in backward
        # function matches the number of Tensors in forward's
        # input args
        if use_bias:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight


def linear_with_grad_accumulation_and_async_allreduce(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[paddle.Tensor] | None = None,
    wgrad_deferral_limit: int | None = 0,
    async_grad_allreduce: bool | None = None,
    tp_group: paddle.core.ProcessGroup | None = None,
    use_accuracy_compatible: bool = False,
    fp8: bool = False,
    fp8_wgrad: bool = False,
    inp_quant_func=None,
    weight_quant_func=None,
    use_pow2_scale: bool = False,
    use_ue8m0: bool = False,
    save_original_input: bool = False,
) -> paddle.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calculation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (paddle.Tensor required): input like paddle.nn.functional.linear

        weight (paddle.Tensor required): weight like paddle.nn.functional.linear

        bias (paddle.Tensor optional): bias like paddle.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        allreduce_dgrad (bool required): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        tp_group (paddle.core.ProcessGroup required): The process group to use for tensor
                                                   parallel operations.

        grad_output_buffer (List[paddle.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Disable by setting this to 0. Defaults to 0.

        async_grad_allreduce (bool optional): Will be removed with 0.11.0.
                                            Please use allreduce_dgrad instead.
    """

    if async_grad_allreduce is not None:
        warnings.warn(
            "async_grad_allreduce is deprecated, not in use anymore and will"
            " be fully removed with 0.11.0. Please use allreduce_dgrad instead."
        )

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        use_accuracy_compatible,
        fp8,
        fp8_wgrad,
        inp_quant_func,
        weight_quant_func,
        use_pow2_scale,
        use_ue8m0,
        save_original_input,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


class Linear(paddle.nn.Layer):
    """Linear layer with no tensor parallelism (weight duplicated across TP ranks).

    The linear layer is defined as Y = XA + b. Weight is not split and is
    replicated on all tensor parallel ranks. Interface is identical to
    ColumnParallelLinear for drop-in compatibility.

    Refer to Megatron-LM's TELinear with parallel_mode="duplicated" for the
    equivalent design.

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias.
        gather_output:
            Unused. Kept for interface compatibility with ColumnParallelLinear.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by
            the caller. This enables performance optimizations where bias can be
            fused with other elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed as a
            keyword argument `weight` during the forward pass. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute
            is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute
            is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object.
        tp_comm_buffer_name:
            Not used. Kept for interface compatibility.
        disable_grad_reduce:
            Not used. Weight is replicated so no TP grad reduction is needed.
        tp_group:
            Not used. Kept for interface compatibility.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: list[paddle.Tensor] | None = None,
        grad_output_buffer: list[paddle.Tensor] | None = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        disable_grad_reduce: bool = False,
        tp_group: paddle.core.ProcessGroup | None = None,
        disable_fp8: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self._dtype = config.params_dtype
        self.tp_comm_buffer_name = tp_comm_buffer_name
        self._fp8_linear_logged = False

        # No TP: output_size_per_partition equals the full output_size.
        self.output_size_per_partition = output_size

        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if config.perform_initialization:
                    _initialize_affine_weight_cpu(
                        self.weight,
                        self.input_size,
                        self.output_size,
                        self.output_size,  # full output, no partition
                        1,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=0,
                        world_size=1,
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )

            # Weight is duplicated across TP ranks; reduce gradient on DP group.
            self.weight.allreduce = True
            self.weight.is_distributed = False
            self.weight.is_expert_param = self.is_expert
            self._mark_replicated_grad_needs_tp_reduction(self.weight)
        else:
            self.weight = None

        if bias:
            self.bias = self.create_parameter(
                shape=[self.output_size],
                dtype=config.params_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                with paddle.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = True
            self.bias.is_distributed = False
            self._mark_replicated_grad_needs_tp_reduction(self.bias)
        else:
            self.bias = None

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # FP8 is inherited from ``config`` unless the caller opts out via
        # ``disable_fp8``. ``save_original_input`` defaults to ``not self.fp8``
        # and is a settable attribute — callers can override it post-init when
        # the wgrad path needs the bf16 activation kept (e.g. the shared
        # expert's up_gate_proj).
        self.fp8 = (
            bool(getattr(config, "full_fp8_computation", False))
            and bool(getattr(config, "fp8", None))
            and not disable_fp8
        )
        self.fp8_wgrad = bool(getattr(config, "fp8_wgrad", False)) and self.fp8
        self.save_original_input = not self.fp8
        self.use_pow2_scale = False
        self.use_ue8m0 = False
        self.inp_quant_func = None
        self.weight_quant_func = None
        if self.fp8:
            # Linear has no TP; the assertion is trivially satisfied but kept
            # for symmetry with ColumnParallelLinear / RowParallelLinear.
            from paddlefleet.fp8.quantization import get_quant_func

            self.use_pow2_scale = (
                paddle.device.cuda.get_device_capability()[0] == 10
            )
            self.use_ue8m0 = bool(getattr(config, "use_ue8m0", False))
            self.inp_quant_func, self.weight_quant_func = get_quant_func(
                getattr(config, "fp8_recipe", "blockwise"),
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=self.use_pow2_scale,
                use_ue8m0=self.use_ue8m0,
            )
            # Color the bf16 weight so ``clear_param_storage("linear_fp8")``
            # can free it after pre-quant.
            _maybe_color_linear_fp8_weight(self)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        """Pre-quantize this Linear's weight and cache on ``self.weight``.

        Called from the per-step FP8 quant callback via
        ``TransformerLayer.fp8_quant_weight`` -> non-MoE Linear walk. Stores
        one fp8 tensor (forward orientation) plus both scales; backward fp8
        is derived on demand in ``weight_quant_func`` via ``.T.contiguous()``.

        ``batch_mode`` / ``quant_transpose`` are accepted for signature
        compatibility with the MoE dispatch chain but are unused here — a
        non-MoE Linear has a single weight tensor and always needs both
        scale orientations (no reason to skip the transposed scale).
        """
        _fp8_prequant_weight(self)

    def clear_fp8_quant_weight(self):
        """Drop the fp8 cache stashed by :meth:`fp8_quant_weight`.

        Must be called after every optimizer step (paired with
        ``fp8_quant_weight`` on the next iter) so a stale post-step weight
        cache does not shadow the freshly-updated bf16 weight in
        ``weight_quant_func``.
        """
        _fp8_clear_prequant_weight(self)

    def forward(
        self,
        input_: paddle.Tensor,
        weight: paddle.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ):
        """Forward of Linear (no tensor parallelism).

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden].
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Unused. Kept for interface compatibility.

        Returns:
            - output
            - bias
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to Linear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            expected_shape = [self.input_size, self.output_size]
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer)
                < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_)

        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = (
                linear_with_grad_accumulation_and_async_allreduce
            )

        output = self._forward_impl(
            input=input_,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            grad_output_buffer=(
                self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            tp_group=None,
            use_accuracy_compatible=getattr(
                self.config, "use_accuracy_compatible", False
            ),
            fp8=self.fp8,
            fp8_wgrad=self.fp8_wgrad,
            inp_quant_func=self.inp_quant_func,
            weight_quant_func=self.weight_quant_func,
            use_pow2_scale=self.use_pow2_scale,
            use_ue8m0=self.use_ue8m0,
            save_original_input=self.save_original_input,
        )

        output_bias = (
            self.bias.clone()
            if (self.skip_bias_add and self.bias is not None)
            else None
        )

        return output, output_bias

    def _mark_replicated_grad_needs_tp_reduction(self, parameter) -> None:
        """Mark a replicated parameter so its gradient is reduced over the TP group.

        The weight is duplicated across TP ranks, but under sequence parallelism
        each rank only sees ``s / TP`` of the sequence, so the local wgrad is a
        PARTIAL sum over the sequence dimension. The full gradient is the sum over
        the TP group. ``Linear.forward`` deliberately passes
        ``sequence_parallel=False`` into the autograd function (this layer never
        gathers or scatters), so nothing else adds that term.

        Paddle's transport for it is the ``sequence_parallel`` attribute:
        ``mark_as_sequence_parallel_parameter`` sets it, and
        ``SPGradSyncCallback`` (PaddleFormers ``trainer/trainer_callback.py``)
        all-reduces exactly the marked parameters over the model-parallel group
        with ``scale=1.0`` (sum, not mean) at ``on_optimizer_begin``.

        This mirrors Megatron-Core, which does the same for the duplicated case in
        ``megatron/core/extensions/transformer_engine.py:930-935``::

            if parallel_mode == "duplicated":
                setattr(param, "sequence_parallel", self.config.sequence_parallel)
                setattr(param, "tensor_model_parallel", False)

        whose gradients ``distributed/finalize_model_grads.py:408`` then all-reduces
        over the TP group. It also mirrors PaddleFormers' own deepseek_v3
        (``transformers/deepseek_v3/modeling.py:651-656``), which marks precisely
        its replicated ``q_a_proj`` / ``kv_a_proj_with_mqa`` this way.

        Expert parameters are excluded: their gradients live in their own
        (expert-)data-parallel domain and are reduced by that path instead, which is
        also why mcore only takes the duplicated branch for non-expert parameters.
        """
        if self.is_expert:
            return
        if not getattr(self.config, "sequence_parallel", False):
            return
        if getattr(self.config, "tensor_model_parallel_size", 1) <= 1:
            return
        mark_as_sequence_parallel_parameter(parameter)

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Weight is replicated, no sharding rules needed."""
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(
            state_dict, None, structured_name_prefix
        )

    def set_extra_state(self, state):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def __repr__(self):
        use_bias = self.bias is not None
        return (
            f"{type(self).__name__}(in_features={self.input_size}, "
            f"out_features={self.output_size}, bias={use_bias}, TP=1)"
        )


def column_sequence_parallel_linear(
    x,
    weight,
    bias=None,
    mp_group=None,
):
    """Functional version of ColumnSequenceParallelLinear using fused_linear.

    Forward: all-gather input along seq dim -> fused_linear.
    Input shape: [seq/mp, batch, hidden], output shape: [seq, batch, hidden/mp].

    Args:
        x: Input tensor (sequence-parallel partitioned).
        weight: Weight tensor, shape [in_features, out_features_per_partition].
        bias: Bias tensor, shape [out_features_per_partition], or None.
        mp_group: Tensor parallel process group.

    Returns:
        Output tensor.
    """
    from paddle.incubate.nn.functional import fused_linear

    from paddlefleet.tensor_parallel.sequence_parallel_utils_legacy import (
        AllGatherOpLegacy,
    )

    is_mp = mp_group is not None and mp_group.nranks > 1

    if is_mp:
        input_parallel = AllGatherOpLegacy.apply(x, 0, mp_group)
    else:
        input_parallel = x

    output = fused_linear(input_parallel, weight, bias)
    return output


def row_sequence_parallel_linear(
    x,
    weight,
    bias=None,
    mp_group=None,
):
    """Functional version of RowSequenceParallelLinear using fused_linear.

    Forward: fused_linear -> reduce-scatter along seq dim.
    Input shape: [seq, batch, hidden/mp], output shape: [seq/mp, batch, hidden].

    Args:
        x: Input tensor (already column-parallel partitioned).
        weight: Weight tensor, shape [in_features_per_partition, out_features].
        bias: Bias tensor, shape [out_features], or None.
        mp_group: Tensor parallel process group.

    Returns:
        Output tensor.
    """
    from paddle.incubate.nn.functional import fused_linear

    from paddlefleet.tensor_parallel.sequence_parallel_utils_legacy import (
        ReduceScatterOpLegacy,
    )

    is_mp = mp_group is not None and mp_group.nranks > 1

    if is_mp:
        output_parallel = fused_linear(x, weight, None)
        output_ = ReduceScatterOpLegacy.apply(output_parallel, mp_group)
        if bias is not None:
            output = output_ + bias
        else:
            output = output_
    else:
        output = fused_linear(x, weight, bias)

    return output


def vocab_parallel_embedding(
    input_,
    weight,
    vocab_start_index,
    num_embeddings,
    mp_group=None,
):
    """Functional version of VocabParallelEmbedding.

    Uses _c_lookup_table + _mp_allreduce (same as paddle fleet mp_layers).

    Args:
        input_: Input token ids tensor.
        weight: Embedding weight, shape [num_embeddings_per_partition, embedding_dim].
        vocab_start_index: Start index of the vocab partition on this rank.
        num_embeddings: Total vocabulary size.
        mp_group: Tensor parallel process group.

    Returns:
        Output embedding tensor.
    """
    from paddle.distributed.fleet.layers.mpu import mp_ops

    is_mp = mp_group is not None and mp_group.nranks > 1

    if is_mp:
        output_parallel = mp_ops._c_lookup_table(
            weight,
            input_,
            start_index=vocab_start_index,
            vocab_size=num_embeddings,
        )
        output = mp_ops._mp_allreduce(
            output_parallel,
            group=mp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )
    else:
        output = F.embedding(input_, weight=weight)

    return output


class ColumnParallelLinear(paddle.nn.Layer):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias
        gather_output:
            If true, call all-gather on output and make Y available to all GPUs,
            otherwise, every GPU will have its output which is Y_i = XA_i
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimations where bias can be fused with other
            elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed
            as a keyword argument `weight` during the forward pass. Note that this does not
            affect bias, which will be allocated if bias is True. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding
            linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object
        tp_comm_buffer_name:
            Communication buffer name is not used in non-Transformer-Engine modules.
        disable_grad_reduce:
            If True, reduction of output gradients across tensor-parallel ranks
            will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to
            delay and fuse reduction along with other gradients for performance optimization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: list[paddle.Tensor] | None = None,
        grad_output_buffer: list[paddle.Tensor] | None = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,  # Not used
        disable_grad_reduce: bool = False,
        tp_group: paddle.core.ProcessGroup | None = None,
        disable_fp8: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce
        self.tp_group = tp_group
        self._dtype = config.params_dtype
        self.tp_comm_buffer_name = tp_comm_buffer_name
        self._fp8_linear_logged = False

        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert, check_initialized=False
        )
        self.world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)
        self.rank = rank
        self.explicit_expert_comm = self.is_expert and (
            self.world_size > 1 or self.expert_parallel
        )
        self.output_size_per_partition = divide(output_size, self.world_size)

        # Parameters.
        # Initialize weight.
        # Note: create the transpose weight, in linear function, the weight
        # should be transposed.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size_per_partition],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )

                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.input_size,
                        self.output_size,
                        self.output_size_per_partition,
                        1,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=self.world_size,
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size_per_partition],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )

            self.weight.allreduce = not (
                self.is_expert and self.expert_parallel
            )
            self.weight.is_distributed = True if self.world_size > 1 else False
            self.weight.is_expert_param = self.is_expert
        else:
            self.weight = None

        if bias:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                dtype=config.params_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )

            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with paddle.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = not (self.is_expert and self.expert_parallel)
            self.bias.is_distributed = True if self.world_size > 1 else False
        else:
            self.bias = None
            # self.register_parameter("bias", None)

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and self.world_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {self.world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = (
            self.world_size > 1
            and not self.sequence_parallel
            and not self.disable_grad_reduce
        )

        self.gradient_accumulation_fusion = False

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # FP8 is inherited from ``config`` unless the caller opts out via
        # ``disable_fp8``. ``save_original_input`` defaults to ``not self.fp8``
        # and is a settable attribute — callers can override it post-init.
        self.fp8 = (
            bool(getattr(config, "full_fp8_computation", False))
            and bool(getattr(config, "fp8", None))
            and not disable_fp8
        )
        self.fp8_wgrad = bool(getattr(config, "fp8_wgrad", False)) and self.fp8
        self.save_original_input = not self.fp8
        self.use_pow2_scale = False
        self.use_ue8m0 = False
        self.inp_quant_func = None
        self.weight_quant_func = None
        if self.fp8:
            assert self.world_size == 1, (
                "ColumnParallelLinear FP8 currently requires TP=1, "
                f"got world_size={self.world_size}"
            )
            from paddlefleet.fp8.quantization import get_quant_func

            self.use_pow2_scale = (
                paddle.device.cuda.get_device_capability()[0] == 10
            )
            self.use_ue8m0 = bool(getattr(config, "use_ue8m0", False))
            self.inp_quant_func, self.weight_quant_func = get_quant_func(
                getattr(config, "fp8_recipe", "blockwise"),
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=self.use_pow2_scale,
                use_ue8m0=self.use_ue8m0,
            )
            # Color the bf16 weight so ``clear_param_storage("linear_fp8")``
            # can free it after pre-quant.
            _maybe_color_linear_fp8_weight(self)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        """Pre-quantize this ColumnParallelLinear's weight (see ``Linear.fp8_quant_weight``)."""
        _fp8_prequant_weight(self)

    def clear_fp8_quant_weight(self):
        """Drop the fp8 cache (see ``Linear.clear_fp8_quant_weight``)."""
        _fp8_clear_prequant_weight(self)

    def forward(
        self,
        input_: paddle.Tensor,
        weight: paddle.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ):
        """Forward of ColumnParallelLinear

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias

        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = [self.input_size, self.output_size_per_partition]
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if getattr(self.config, "gpt_model_use_experimental_version", False):
            output = column_sequence_parallel_linear(
                input_, weight, self.bias, mp_group=self.tp_group
            )
            return output, None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
            or (self.tp_group is not None and self.tp_group.world_size == -1)
            or self.tp_group is None
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(
                input_,
                group=self.tp_group,
                is_expert=self.is_expert,
            )

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer)
                < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = (
                linear_with_grad_accumulation_and_async_allreduce
            )

        allreduce_dgrad = (
            False if self.explicit_expert_comm else self.allreduce_dgrad
        )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert self.config.cpu_offloading is False, (
                        "CPU Offloading cannot be enabled while TE is not present"
                    )
                else:
                    input_parallel.activation_offloading = (
                        self.config.cpu_offloading_activations
                    )

        _colpar_acc = (
            os.environ.get("MODEL_REPRO_COLPAR_ACC", "0") == "1"
            and get_pg_size(self.tp_group) <= 1
        )
        if _colpar_acc:
            if os.environ.get("MODEL_REPRO_MOE_TN_GEMM", "0") == "1":
                # E-107: TN layout — matmul(x, w.t().contiguous(), transpose_y=True)
                # is bit-exact with mcore's matmul(x, w.t()) (e107 probes).
                output_parallel = paddle.matmul(
                    input_parallel.contiguous(),
                    weight.t().contiguous(),
                    transpose_y=True,
                )
            else:
                # E-101 probe: for non-parallel column linears (e.g. EP1/ETP1 MoE
                # expert up_gate_proj) run a plain contiguous F.linear so the bf16
                # rounding matches the torch eager/TE path, mirroring the o_proj fix.
                output_parallel = F.linear(
                    input_parallel.contiguous(), weight, bias
                )
        else:
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=allreduce_dgrad,
                sequence_parallel=False
                if self.explicit_expert_comm
                else self.sequence_parallel,
                grad_output_buffer=(
                    self.grad_output_buffer
                    if self.config.defer_embedding_wgrad_compute
                    else None
                ),
                wgrad_deferral_limit=(
                    self.config.wgrad_deferral_limit
                    if self.config.defer_embedding_wgrad_compute
                    else None
                ),
                tp_group=self.tp_group,
                use_accuracy_compatible=getattr(
                    self.config, "use_accuracy_compatible", False
                ),
                fp8=self.fp8,
                fp8_wgrad=self.fp8_wgrad,
                inp_quant_func=self.inp_quant_func,
                weight_quant_func=self.weight_quant_func,
                use_pow2_scale=self.use_pow2_scale,
                use_ue8m0=self.use_ue8m0,
                save_original_input=self.save_original_input,
            )

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output = output_parallel
        output_bias = (
            self.bias.clone()
            if (self.skip_bias_add and self.bias is not None)
            else None
        )

        return output, output_bias

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 1, bias sharded"""
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 1, "bias": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )

    def set_extra_state(self, state):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def __repr__(self):
        tp = self.output_size // self.output_size_per_partition
        use_bias = self.bias is not None
        return (
            f"{type(self).__name__}(in_features={self.input_size}, "
            f"out_features={self.output_size}, bias={use_bias}, TP={tp})"
        )


class RowParallelLinear(paddle.nn.Layer):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias. Note that bias is not parallelized.
        input_is_parallel:
            If true, we assume that the input is already split across the GPUs
            and we do not split again.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It returns the master weights
            used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimations where bias can be fused with other
            elementwise operations.
        is_expert:
            If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name:
            Communication buffer name. Not used in non-Transformer-Engine modules.
        config:
            FleetConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,  # Not used
        tp_group: paddle.core.ProcessGroup | None = None,
        disable_fp8: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        # self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.gradient_accumulation_fusion = False
        self.sequence_parallel = config.sequence_parallel
        self.tp_group = tp_group
        self._dtype = config.params_dtype
        self.tp_comm_buffer_name = tp_comm_buffer_name
        self._fp8_linear_logged = False

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_parallel`, `input_is_parallel` must be `True`"
            )

        # Divide the weight matrix along the last dimension.
        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert, check_initialized=False
        )

        self.world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)
        self.explicit_expert_comm = self.is_expert and (
            self.world_size > 1 or self.expert_parallel
        )

        self.input_size_per_partition = divide(input_size, self.world_size)

        # Parameters.
        # Note: create the transposed weight here, and the weight should
        # be transposed back in the forward function of linear.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.output_size],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.input_size,
                    self.output_size,
                    self.input_size_per_partition,
                    0,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=self.world_size,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.output_size],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    is_expert=self.is_expert,
                )
        self.weight.allreduce = not (self.is_expert and self.expert_parallel)
        self.weight.is_distributed = True if self.world_size > 1 else False
        self.weight.is_expert_param = self.is_expert

        if bias:
            self.bias = self.create_parameter(
                shape=[self.output_size],
                dtype=config.params_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with paddle.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = not (self.is_expert and self.expert_parallel)
            self.bias.sequence_parallel = self.sequence_parallel
        else:
            self.bias = None
            # self.register_parameter("bias", None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # FP8 is inherited from ``config`` unless the caller opts out via
        # ``disable_fp8``. ``save_original_input`` defaults to ``not self.fp8``
        # and is a settable attribute — callers can override it post-init.
        self.fp8 = (
            bool(getattr(config, "full_fp8_computation", False))
            and bool(getattr(config, "fp8", None))
            and not disable_fp8
        )
        self.fp8_wgrad = bool(getattr(config, "fp8_wgrad", False)) and self.fp8
        self.save_original_input = not self.fp8
        self.use_pow2_scale = False
        self.use_ue8m0 = False
        self.inp_quant_func = None
        self.weight_quant_func = None
        if self.fp8:
            assert self.world_size == 1, (
                "RowParallelLinear FP8 currently requires TP=1, "
                f"got world_size={self.world_size}"
            )
            from paddlefleet.fp8.quantization import get_quant_func

            self.use_pow2_scale = (
                paddle.device.cuda.get_device_capability()[0] == 10
            )
            self.use_ue8m0 = bool(getattr(config, "use_ue8m0", False))
            self.inp_quant_func, self.weight_quant_func = get_quant_func(
                getattr(config, "fp8_recipe", "blockwise"),
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=self.use_pow2_scale,
                use_ue8m0=self.use_ue8m0,
            )
            # Color the bf16 weight so ``clear_param_storage("linear_fp8")``
            # can free it after pre-quant.
            _maybe_color_linear_fp8_weight(self)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        """Pre-quantize this RowParallelLinear's weight (see ``Linear.fp8_quant_weight``)."""
        _fp8_prequant_weight(self)

    def clear_fp8_quant_weight(self):
        """Drop the fp8 cache (see ``Linear.clear_fp8_quant_weight``)."""
        _fp8_clear_prequant_weight(self)

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        if getattr(self.config, "gpt_model_use_experimental_version", False):
            output = row_sequence_parallel_linear(
                input_, self.weight, self.bias, mp_group=self.tp_group
            )
            return output, None

        # Set up backprop all-reduce.
        if (
            self.input_is_parallel
            or self.tp_group is None
            or (self.tp_group is not None and self.tp_group.nranks == 1)
        ):
            # NOTE: if tp_group only contains one rank, directly set input_parallel to input_
            # otherwise it will fail in scatter_to_tensor_model_parallel_region.
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )

        # Matrix multiply.
        if not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = (
                linear_with_grad_accumulation_and_async_allreduce
            )

        allreduce_dgrad = False

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert self.config.cpu_offloading is False, (
                        "CPU Offloading cannot be enabled while TE is not present"
                    )
                else:
                    input_parallel.activation_offloading = (
                        self.config.cpu_offloading_activations
                    )

        if os.environ.get("MODEL_REPRO_OPROJ_ACC", "0") == "1":
            if os.environ.get("MODEL_REPRO_MOE_TN_GEMM", "0") == "1":
                # E-107: TN layout — matmul(x, w.t().contiguous(), transpose_y=True)
                # is bit-exact with mcore's matmul(x, w.t()) (e107 probes).
                output_parallel = paddle.matmul(
                    input_parallel.contiguous(),
                    self.weight.t().contiguous(),
                    transpose_y=True,
                )
            else:
                # E-091 probe: run the local row-parallel GEMM as a
                # plain contiguous F.linear so the bf16 rounding matches torch's TE Linear.
                output_parallel = F.linear(
                    input_parallel.contiguous(), self.weight, None
                )
        else:
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=None,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=allreduce_dgrad,
                sequence_parallel=False,
                tp_group=None,
                grad_output_buffer=None,
                use_accuracy_compatible=getattr(
                    self.config, "use_accuracy_compatible", False
                ),
                fp8=self.fp8,
                fp8_wgrad=self.fp8_wgrad,
                inp_quant_func=self.inp_quant_func,
                weight_quant_func=self.weight_quant_func,
                use_pow2_scale=self.use_pow2_scale,
                use_ue8m0=self.use_ue8m0,
                save_original_input=self.save_original_input,
            )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            if os.environ.get("MODEL_REPRO_ROWPAR_FP32", "0") == "1":
                # E-090 probe: do the TP reduce-scatter in fp32 so the row-parallel
                # reduction order/precision matches the torch TE path, then cast back.
                _odt = output_parallel.dtype
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_parallel.cast("float32"), group=self.tp_group
                ).cast(_odt)
            else:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_parallel, group=self.tp_group
                )
        else:
            output_ = reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group, is_expert=self.is_expert
            )
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias.clone() if self.bias is not None else None
        return output, output_bias

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )

    def set_extra_state(self, state):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def __repr__(self):
        tp = self.input_size // self.input_size_per_partition
        use_bias = self.bias is not None
        return (
            f"{type(self).__name__}(in_features={self.input_size}, "
            f"out_features={self.output_size}, bias={use_bias}, TP={tp})"
        )
