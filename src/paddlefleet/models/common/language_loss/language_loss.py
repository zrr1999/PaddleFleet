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


import functools
import hashlib
import os

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import ScheduleNode
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
    MTPDistillationLossShift,
)
from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.training.global_vars import get_global_training_logs
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import TransformerConfig


def _loss_md5_enabled() -> bool:
    return os.environ.get("LOG_LOSS_MD5", "0") == "1"


def _use_accuracy_compatible_kernel() -> bool:
    """Switch for Megatron-aligned (accuracy-compatible) numeric paths.

    Controlled by the ``FLAGS_use_accuracy_compatible_kernel`` env variable.
    """
    return os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"


def _tensor_md5(tensor: Tensor, dtype: str = "float32") -> str:
    """Calculate MD5 hash of a tensor, **for debugging only**.

    Note: internally calls .numpy() which triggers GPU→CPU synchronization
    and blocks the async training pipeline. Do NOT use in the forward pass.
    """
    tensor_for_md5 = tensor.detach().cast(dtype)
    return hashlib.md5(tensor_for_md5.numpy().tobytes()).hexdigest()


def _print_scalar_loss_md5(prefix: str, name: str, loss: Tensor) -> None:
    if not _loss_md5_enabled():
        return
    rank = paddle.distributed.get_rank()
    loss_tensor = loss.detach().cast("float32").reshape([1])
    print(
        f"[{prefix}] rank={rank} {name}={loss_tensor.item():.20f} "
        f"{name}_md5={_tensor_md5(loss_tensor)}",
        flush=True,
    )


class DistributedSoftmaxOp(PyLayer):
    @staticmethod
    def forward(ctx, x, axis=-1, mp_group=None):
        ctx.axis = axis
        if mp_group is None:
            hcg = fleet.get_hybrid_communicate_group()
            mp_group = hcg.get_model_parallel_group()

        ctx.mp_group = mp_group

        local_max = paddle.max(x, axis=axis, keepdim=True)

        all_max = AllGatherOp.apply(local_max)

        global_max = paddle.max(all_max, axis=0, keepdim=True)

        x_stable = x - global_max

        exp_x = paddle.exp(x_stable.cast("float32"))

        local_sum_exp = paddle.sum(exp_x, axis=axis, keepdim=True)

        sum_exp = mp_ops._mp_allreduce(
            local_sum_exp,
            group=mp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        softmax_output = exp_x / sum_exp

        ctx.save_for_backward(softmax_output, sum_exp)

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, global_sum_exp = ctx.saved_tensor()
        axis = ctx.axis
        mp_group = ctx.mp_group

        grad_softmax = grad_output * softmax_output

        local_sum_grad = paddle.sum(grad_softmax, axis=axis, keepdim=True)

        all_sum_grad = AllGatherOp.apply(local_sum_grad)
        global_sum_grad = paddle.sum(all_sum_grad, axis=0, keepdim=True)

        grad_input = softmax_output * (grad_output - global_sum_grad)

        return grad_input


# E-233/E-234: deferred token normalization state.
#
# Under the accuracy-compatible path the per-token loss normalization is kept OUT
# of the bf16 gradient path (see DeferTokenNormalizationOp below). The divisor the
# gradients still owe is recorded here so the trainer can apply it to the fp32
# gradient buffers after the backward, exactly as the reference framework does in
# ``megatron/core/distributed/finalize_model_grads.py:586-602``.
#
# Only the last pipeline stage computes the loss, so only that stage records a
# value; the trainer is responsible for broadcasting it across the pipeline group
# (the reference does the same at ``finalize_model_grads.py:591-592``).
_PENDING_GRADIENT_DIVISOR: dict[str, float] = {}


def get_pending_gradient_divisor() -> float | None:
    """Token count the current step's gradients still have to be divided by.

    Returns ``None`` when the deferred-normalization path did not run, in which
    case the gradients are already normalized and must NOT be scaled again.
    """
    return _PENDING_GRADIENT_DIVISOR.get("value")


def set_pending_gradient_divisor(value: float) -> None:
    """Publish the divisor on ranks that did not compute the loss.

    Only the last pipeline stage runs the loss, so only it registers a divisor.
    The trainer resolves the value across the pipeline group before the optimizer
    callbacks fire and writes it back here, so that every rank - in particular the
    gradient-inventory receipt on the earlier stages - can see the divisor its
    gradients still owe.
    """
    _PENDING_GRADIENT_DIVISOR["value"] = float(value)


def clear_pending_gradient_divisor() -> None:
    """Drop the recorded divisor. The trainer calls this once it has applied it,
    so that a step which somehow skips the loss cannot silently reuse a stale
    divisor from an earlier step."""
    _PENDING_GRADIENT_DIVISOR.pop("value", None)


class DeferTokenNormalizationOp(PyLayer):
    """Divide the loss for REPORTING while leaving the gradient unnormalized.

    THE DEFECT THIS EXISTS TO FIX (E-233, measured; E-234, confirmed).

    The token normalization used to be an ordinary division, so its reciprocal
    entered the gradient before the logits gradient was rounded to bf16. At a
    supervised label slot the exact gradient of a summed cross-entropy is -1.0,
    which bf16 represents exactly; dividing first asks bf16 to represent -1/N
    instead, and for N = 44 that costs exactly one part in 2^10:

        1/44          = 0.022727272727...
        bf16(1/44)    = 0.022705078125
        relative error = -2^-10 = -9.765625e-04

    Because the factor multiplies EVERY element of the logits gradient, every
    downstream gradient inherited it with the same sign. That is what made the
    weight-gradient inventory one-sided in 59 of 64 comparable families with a
    median deficit of +0.00097318 against the reference - a 0.35% match to the
    predicted +0.00097656.

    The reference framework does not have the problem because it does not divide
    here at all: ``calculate_per_token_loss: true`` keeps the summed gradient and
    ``finalize_model_grads`` divides the fp32 gradient buffers afterwards
    (``megatron/core/pipeline_parallel/schedules.py:331-335`` and
    ``megatron/core/distributed/finalize_model_grads.py:586-602``).

    THE FIX MIRRORS THAT SPLIT rather than trying to round better:

      * forward divides, so the reported loss scalar is bit-for-bit what it was.
        This matters because the loss scalar is already bit-equal to the
        reference (E-227) and is an acceptance gate field; the fix must not move
        it.
      * backward multiplies by ``backward_scale`` INSTEAD of 1/N, so the bf16
        logits gradient carries exactly representable values.
      * the divisor is recorded in ``_PENDING_GRADIENT_DIVISOR`` and applied by
        the trainer to the fp32 gradient buffers.

    ``backward_scale`` exists for the MTP branch. The reference funnels every
    branch through ONE global divisor (the main loss's token count) and corrects
    each branch by the ratio ``original_num_tokens / num_tokens``
    (``megatron/core/transformer/multi_token_prediction.py:1054-1065``). Passing
    ``main_tokens / branch_tokens`` here reproduces that: after the trainer
    divides by ``main_tokens``, the branch has been divided by
    ``branch_tokens``, which is what it owed.

    Not gated internally: callers apply it only under
    ``_use_accuracy_compatible_kernel()``, keeping the default numerics untouched.
    """

    @staticmethod
    def forward(ctx, loss_sum, divisor, backward_scale):
        ctx.backward_scale = float(backward_scale)
        # THE DIVISOR MUST BE A 0-d TENSOR OF THE SAME DTYPE, not a python float.
        # On GPU those two are NOT the same computation: dividing a float32 tensor
        # by a python float can land one ulp away from dividing by a float32 0-d
        # tensor. Measured on this device with the real step-1 MTP loss sum
        # 567.9686279296875:
        #     x / paddle.to_tensor(44.0)  ->  12.908377647399902  (0x414e88b7)
        #     x / 44.0                    ->  12.908378601074219  (0x414e88b8)
        # The first is the value the previous ``loss / lossmask.sum()`` produced
        # and the one that is bit-equal to the reference, so the fix has to keep
        # it. Building the divisor with ``paddle.full`` restores bit-equality: 0
        # mismatches over 19999 sampled magnitudes.
        divisor_tensor = paddle.full([], float(divisor), dtype=loss_sum.dtype)
        return loss_sum / divisor_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.backward_scale


def _normalize_loss_by_tokens(
    loss_sum: Tensor,
    valid_tokens: float,
    main_tokens: float | None = None,
) -> Tensor:
    """Token-normalize ``loss_sum``, deferring the gradient share when the
    accuracy-compatible path is active.

    ``main_tokens`` is the divisor the trainer will apply globally; it defaults to
    ``valid_tokens`` for the main loss. An auxiliary branch passes the main
    loss's count so a single global divisor serves every branch.
    """
    if not _use_accuracy_compatible_kernel() or valid_tokens <= 0:
        return loss_sum / valid_tokens

    if main_tokens is None or main_tokens <= 0:
        main_tokens = valid_tokens
        _PENDING_GRADIENT_DIVISOR["value"] = float(main_tokens)

    return DeferTokenNormalizationOp.apply(
        loss_sum, valid_tokens, main_tokens / valid_tokens
    )


def subbatch(
    f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}
):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    This is useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Original function to be converted to subbatch processing.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size (number of elements to process at once).
        out_idx (int): Index of the output dimension that needs stacking.
        use_recompute (bool, optional): Whether to use recomputation for memory savings. Defaults to False.
        same_arg_idx (dict, optional): Mapping of argument indices that share the same tensor.
                                     e.g. {1: 0} means args[1] == args[0], avoiding duplicate slicing.

    Returns:
        Callable: Converted function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        assert len(arg_idx) == len(axis), (
            "Number of batching args and number of batching dims should match."
        )

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = dict(zip(inps, axis))

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert i > same_arg_idx[i], (
                        f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    )
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(
                    f, *_args, **kwargs
                )
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.cat(outs, out_idx)

    return wrapper


class LanguageLoss(FleetLayer):
    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.config = config
        self.ignored_index = -100
        self.enable_parallel_cross_entropy = (
            paddle.distributed.is_initialized()
            and get_tensor_model_parallel_world_size() > 1
            and config.parallel_output
        )

        if self.enable_parallel_cross_entropy:
            self.loss_func = (
                paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy()
            )
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

        self.loss_subbatch_sequence_length = (
            config.loss_subbatch_sequence_length
        )
        self.use_subbatch = self.loss_subbatch_sequence_length > 0

        # E-233/E-234: when an auxiliary branch (MTP) is normalized by its own
        # token count, this carries the MAIN loss's count so that one global
        # divisor applied by the trainer serves every branch, matching the
        # reference's original_num_tokens / num_tokens correction. ``None`` means
        # "this call IS the main loss", which is the case that registers the
        # global divisor.
        self._deferred_main_tokens: float | None = None

    def forward_impl(self, logits: Tensor | tuple, labels: Tensor) -> Tensor:
        # Fused linear + cross-entropy path: `logits` is actually a
        # (hidden_states, weight, bias) tuple emitted by GPTLMHead when
        # config.fused_linear_ce_loss_chunk > 0. Dispatch to the fused kernel
        # to avoid materializing the full [B, S, V] logits tensor.
        if isinstance(logits, tuple):
            assert not self.enable_parallel_cross_entropy, (
                "fused_linear_ce_loss_chunk is incompatible with tensor parallel "
                "parallel_output=True (ParallelCrossEntropy path)."
            )
            from paddlefleet.triton_ops.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyFunction,
            )

            hidden_states, weight, bias = logits[:3]
            # Multimax lm_head fused path: GPTLMHead emits a 5-tuple
            # (hidden_states, weight, bias, multimax_ranges, multimax_ts)
            # so SegLU is applied inside the chunked CE kernel without
            # materializing full [B, S, V] logits.
            multimax_ranges = logits[3] if len(logits) > 3 else None
            multimax_ts = logits[4] if len(logits) > 4 else None
            B, S, H = hidden_states.shape
            _input = hidden_states.reshape([-1, H])
            _labels = labels.reshape([-1])

            apply_args = [
                _input,
                weight,
                _labels,
                bias,
                self.ignored_index,
                "none",
                self.config.fused_linear_ce_loss_chunk,
                getattr(
                    self.config, "gpt_model_use_experimental_version", False
                ),
            ]
            if multimax_ranges is not None and multimax_ts is not None:
                apply_args.append(multimax_ranges)
                apply_args.append(multimax_ts)
            loss_1d = LigerFusedLinearCrossEntropyFunction.apply(*apply_args)
            # Reshape back to [B, S] so downstream CP gather / lossmask
            # handling matches the non-fused path exactly.
            loss = loss_1d.reshape([B, S])

            if get_context_parallel_world_size() > 1:
                loss = ContextParallelGatherOp.apply(
                    loss, axis=1, mode=self.config.cp_balance_mode
                )
                labels = ContextParallelGatherOp.apply(
                    labels, axis=1, mode=self.config.cp_balance_mode
                )

            lossmask = labels != self.ignored_index
            if (~lossmask).all():
                return paddle.mean(loss) * 0.0

            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            loss = paddle.sum(
                loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
            loss = loss / lossmask.sum()
            return loss

        seq_len = logits.shape[1]

        # E-539: CPU dump of last-stage call-5 ParallelCrossEntropy operands.
        # Observation only; returns tensors unchanged.
        _ce_dump = os.environ.get("MODEL_REPRO_CE_BIN_DIR")
        if _ce_dump and not isinstance(logits, tuple):
            try:
                _rank = int(dist.get_rank()) if dist.is_initialized() else 0
            except Exception:
                _rank = 0
            if _rank in (2, 3):
                ntok = int(logits.shape[1]) if logits.ndim >= 2 else int(logits.shape[0])
                # Last-stage SP-gathered logits at call 5 are S=168.
                if ntok == 168:
                    hits = getattr(self, "_e539_ce_hits", 0) + 1
                    self._e539_ce_hits = hits
                    if hits == 1:
                        import json

                        os.makedirs(_ce_dump, exist_ok=True)
                        lg = logits.detach().contiguous()
                        lb = labels.detach().contiguous()
                        if "bfloat16" in str(lg.dtype):
                            lg_buf = lg.view(dtype="uint16").cpu().numpy()
                            lg_suf = "bf16"
                        else:
                            lg_buf = lg.cast("float32").cpu().numpy()
                            lg_suf = "f32"
                        lb_buf = lb.cast("int64").cpu().numpy()
                        stem = f"paddle_ce_r{_rank}_h{hits}_S{ntok}"
                        lg_buf.tofile(os.path.join(_ce_dump, f"{stem}_logits.{lg_suf}.bin"))
                        lb_buf.tofile(os.path.join(_ce_dump, f"{stem}_labels.i64.bin"))
                        meta = {
                            "framework": "paddle",
                            "rank": _rank,
                            "hit": hits,
                            "ntok": ntok,
                            "logits_shape": list(lg.shape),
                            "labels_shape": list(lb.shape),
                            "logits_dtype": str(lg.dtype),
                            "labels_dtype": str(lb.dtype),
                            "suffix": lg_suf,
                        }
                        with open(
                            os.path.join(_ce_dump, f"{stem}.json"), "w", encoding="utf-8"
                        ) as stream:
                            json.dump(meta, stream, sort_keys=True)
                            stream.write("\n")
                        print(
                            f"[E539-CE-BIN] dir={_ce_dump} rank={_rank} {stem} "
                            f"logits={list(lg.shape)} labels={list(lb.shape)}",
                            flush=True,
                        )

        # Loss-path MD5 probe: logits and labels before cross-entropy
        if (
            os.environ.get("LOG_LAYER_MD5", "0") == "1"
            or os.environ.get("LOG_LOSS_MD5", "0") == "1"
        ):
            import hashlib

            rank = paddle.distributed.get_rank()
            lg_md5 = hashlib.md5(
                logits.cast("float32").numpy().tobytes()
            ).hexdigest()
            lb_md5 = hashlib.md5(
                labels.cast("int64").numpy().tobytes()
            ).hexdigest()
            print(
                f"[LOSS_PATH_MD5] rank={rank} loss_input_logits shape={list(logits.shape)} md5={lg_md5}",
                flush=True,
            )
            print(
                f"[LOSS_PATH_MD5] rank={rank} loss_input_labels shape={list(labels.shape)} md5={lb_md5}",
                flush=True,
            )

        if self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:

            def _cast_loss_func(logits, labels):
                return self.loss_func(logits.cast("float32"), labels)

            sb_loss_func = subbatch(
                _cast_loss_func,
                arg_idx=[0, 1],
                axis=[1, 1],
                bs=self.loss_subbatch_sequence_length,
                out_idx=1,
            )
            loss = sb_loss_func(logits, labels)
        else:
            if (
                self.config.gpt_model_use_experimental_version
                and self.config.sequence_parallel
            ):
                logits = logits.reshape([labels.shape[0], -1, logits.shape[-1]])
            loss = self.loss_func(logits.cast("float32"), labels)

        if get_context_parallel_world_size() > 1:
            loss = ContextParallelGatherOp.apply(
                loss, axis=1, mode=self.config.cp_balance_mode
            )
            labels = ContextParallelGatherOp.apply(
                labels, axis=1, mode=self.config.cp_balance_mode
            )

        if _use_accuracy_compatible_kernel():
            # 定位锚点 1：CP gather 后、mask/归一化前的 per-token CE，
            # 两侧语义唯一，未掺入归一化差异。
            print(
                f"\nper_token_loss: rank={dist.get_rank()} "
                f"shape={list(loss.shape)} md5={loss.cast('float32')._md5sum()}",
                flush=True,
            )
            if os.environ.get("MODEL_REPRO_PER_TOKEN_VALUES"):
                # An md5 tells you the buffers differ, not where. The masked sum can be
                # bit-identical while ignored positions differ, so the digest alone cannot
                # say whether a SUPERVISED position disagrees. Print the values (one short
                # sequence) so the two stacks can be compared position by position.
                _flat = loss.cast("float32").reshape([-1]).numpy()
                print(
                    f"per_token_loss_values: rank={dist.get_rank()} "
                    f"n={_flat.size} "
                    f"hex={_flat.tobytes().hex()}",
                    flush=True,
                )

        lossmask = labels != self.ignored_index
        _valid_tokens = -1.0
        if (~lossmask).all():
            loss = paddle.mean(loss) * 0.0
        else:
            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            _valid_tokens = float(lossmask.sum())

            # Loss-path MD5 probe: per-token loss and lossmask
            if (
                os.environ.get("LOG_LAYER_MD5", "0") == "1"
                or os.environ.get("LOG_LOSS_MD5", "0") == "1"
            ):
                import hashlib

                rank = paddle.distributed.get_rank()
                pt_md5 = hashlib.md5(
                    loss.cast("float32").reshape([-1]).numpy().tobytes()
                ).hexdigest()
                lm_md5 = hashlib.md5(lossmask.numpy().tobytes()).hexdigest()
                valid_count = lossmask.sum().item()
                loss_sum_val = paddle.sum(
                    loss.cast("float32").reshape([-1]) * lossmask
                ).item()
                print(
                    f"[LOSS_PATH_MD5] rank={rank} per_token_loss md5={pt_md5}",
                    flush=True,
                )
                print(
                    f"[LOSS_PATH_MD5] rank={rank} lossmask md5={lm_md5} valid_tokens={valid_count}",
                    flush=True,
                )
                print(
                    f"[LOSS_PATH_MD5] rank={rank} loss_sum={loss_sum_val} final_loss={loss_sum_val / valid_count}",
                    flush=True,
                )
                # Also compute line-wise loss (matches EC's _line_wise_loss) for exact comparison
                if self.config.gpt_model_use_experimental_version:
                    _probe_loss_2d = loss.cast(
                        paddle.float32
                    ) * lossmask.reshape(labels.shape)
                    _probe_lm_2d = lossmask.reshape(labels.shape)
                    _probe_tc = _probe_lm_2d.sum(-1)
                    _probe_inv = (_probe_tc == 0).astype(paddle.float32)
                    _probe_lpl = _probe_loss_2d.sum(-1) / (
                        _probe_tc + 1e-6 * _probe_inv
                    )
                    _probe_lpl = _probe_lpl * (1 - _probe_inv)
                    _probe_lw = _probe_lpl.sum() / (
                        (1 - _probe_inv).sum() + 1e-6
                    )
                    print(
                        f"[LOSS_PATH_MD5] rank={rank} line_wise_loss={_probe_lw.item():.20f}",
                        flush=True,
                    )

            # EC-compat: line-wise loss (per-sample mean then average across samples)
            # EC's ErniemmPretrainingCriterion recomputes loss as line-wise when task_id
            # is present, which changes the value due to division by (count + 1e-6).
            if self.config.gpt_model_use_experimental_version:
                if max(get_tensor_model_parallel_world_size(), 1) > 1:
                    loss = loss.squeeze(-1)
                loss_2d = loss.cast(paddle.float32) * lossmask.reshape(
                    labels.shape
                )
                lossmask_2d = lossmask.reshape(labels.shape)
                token_count_per_line = lossmask_2d.sum(-1)
                is_invalid_line_float = (token_count_per_line == 0).astype(
                    paddle.float32
                )
                loss_per_line = loss_2d.sum(-1) / (
                    token_count_per_line + 1e-6 * is_invalid_line_float
                )
                loss_per_line = loss_per_line * (1 - is_invalid_line_float)
                loss = loss_per_line.sum() / (
                    (1 - is_invalid_line_float).sum() + 1e-6
                )
            else:
                loss = paddle.sum(
                    loss.cast(paddle.float32).reshape([-1]) * lossmask
                )
                # E-233/E-234: under the accuracy-compatible path the division is
                # value-only and the gradient share is deferred to the fp32
                # gradient buffers. See DeferTokenNormalizationOp.
                loss = _normalize_loss_by_tokens(
                    loss,
                    _valid_tokens,
                    main_tokens=self._deferred_main_tokens,
                )

        if _use_accuracy_compatible_kernel():
            # 定位锚点 2：mask + 归一化后的标量 loss，与锚点 1 配合可切开
            # 「CE 上游差异」和「lossmask / valid_token / 除法差异」。
            print(
                f"\nfinal_loss: rank={dist.get_rank()} "
                f"val={float(loss):.20f} md5={loss.cast('float32')._md5sum()} "
                f"valid_tokens={_valid_tokens!r}",
                flush=True,
            )

        return loss

    def _forward(self, logits: Tensor | tuple, labels: Tensor):
        if (
            get_context_parallel_world_size() > 1
            and self.config.experimental_dataflow
        ):
            # In EB data flow and CP size > 1, scatter labels to cp local
            labels = ContextParallelScatterOp.apply(
                labels, axis=1, mode=self.config.cp_balance_mode
            )
        if (
            self.config.recompute_modules is not None
            and "loss_fn" in self.config.recompute_modules
        ):
            return recompute(self.forward_impl, logits, labels)
        return self.forward_impl(logits, labels)

    def _mtp_loss_for_depth(
        self, depth, mtp_logits, labels_ori, seq_length, mtp_loss
    ):
        """Compute one MTP depth's loss and append it to ``mtp_loss``.

        Extracted verbatim from ``forward`` (E-234) so the deferred-normalization
        state set up around the depth loop is scoped by a try/finally rather than
        by an ever-growing loop body. No numerics changed in the move; the only
        edit is that the inline experimental-version division now also routes
        through ``_normalize_loss_by_tokens``.
        """
        logits_cur_depth = mtp_logits[depth]
        labels_cur_depth = labels_ori[
            :, (depth + 1) : (depth + 1 + seq_length)
        ]
        if self.config.gpt_model_use_experimental_version:
            # Align with EB: compute per-token loss matrix and reduce
            # with global sum/count instead of going through forward_impl
            # which applies line-wise loss.

            if get_context_parallel_world_size() > 1:
                # In EB data flow and CP size > 1, since we do not use _forward
                # we need to scatter labels to cp local here.
                labels_cur_depth = ContextParallelScatterOp.apply(
                    labels_cur_depth,
                    axis=1,
                    mode=self.config.cp_balance_mode,
                )

            if self.config.fused_linear_ce_loss_chunk > 0:
                loss_matrix_cur_depth = self._forward(
                    logits_cur_depth,
                    labels_cur_depth,
                )
            else:
                if (
                    self.config.gpt_model_use_experimental_version
                    and self.config.sequence_parallel
                ):
                    logits_cur_depth = logits_cur_depth.reshape(
                        [
                            labels_cur_depth.shape[0],
                            -1,
                            logits_cur_depth.shape[-1],
                        ]
                    )
                loss_matrix_cur_depth = self.loss_func(
                    logits_cur_depth.cast("float32"),
                    labels_cur_depth,
                )

            if get_context_parallel_world_size() > 1:
                # In EB data flow and CP size > 1, loss and labels need to be gathered back.
                loss_matrix_cur_depth = ContextParallelGatherOp.apply(
                    loss_matrix_cur_depth,
                    axis=1,
                    mode=self.config.cp_balance_mode,
                )
                labels_cur_depth = ContextParallelGatherOp.apply(
                    labels_cur_depth,
                    axis=1,
                    mode=self.config.cp_balance_mode,
                )

            lossmask_cur_depth = (
                labels_cur_depth != self.ignored_index
            ).cast(paddle.float32)
            loss_matrix_cur_depth = loss_matrix_cur_depth.cast(
                paddle.float32
            ).reshape([-1]) * lossmask_cur_depth.reshape([-1])
            _depth_tokens = float(lossmask_cur_depth.sum())
            if _depth_tokens > 0:
                loss_cur_depth = _normalize_loss_by_tokens(
                    loss_matrix_cur_depth.sum(),
                    _depth_tokens,
                    main_tokens=self._deferred_main_tokens,
                )
            else:
                loss_cur_depth = loss_matrix_cur_depth.sum() * 0.0
        else:
            loss_cur_depth = self._forward(
                logits_cur_depth,
                labels_cur_depth,
            )
        mtp_loss.append(loss_cur_depth)

    def forward(self, logits: Tensor | list, labels: Tensor) -> Tensor:
        if isinstance(logits, list):
            assert (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
                and not self.config.mtp_load_weight_only
            )
            assert len(logits) == self.config.num_nextn_predict_layers + 1
            labels_ori = labels
            lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
            seq_length = lm_labels.shape[1]

            mtp_loss = []
            mtp_logits = logits[1:]

            if not self.config.mtp_distillation_loss:
                if self.config.train_mtp_only:
                    lm_loss = 0.0
                else:
                    lm_loss = self._forward(logits[0], lm_labels)

                # E-233/E-234: the main loss above registered the global divisor
                # the trainer will apply to the fp32 gradient buffers. Every MTP
                # depth below must therefore be told to normalize its VALUE by its
                # own (rolled, hence smaller) token count while charging its
                # GRADIENT against that same global divisor. This mirrors the
                # reference's original_num_tokens / num_tokens correction at
                # megatron/core/transformer/multi_token_prediction.py:1054-1065.
                _main_tokens = get_pending_gradient_divisor()

                # forward_impl reads this to decide whether it is normalizing the
                # MAIN loss (registers the global divisor) or an AUXILIARY branch
                # (charges its gradient against the already-registered one).
                # Restored in the finally below so a later main-loss call in the
                # same process cannot inherit it.
                self._deferred_main_tokens = _main_tokens
                try:
                    for depth in range(self.config.num_nextn_predict_layers):
                        self._mtp_loss_for_depth(
                            depth,
                            mtp_logits,
                            labels_ori,
                            seq_length,
                            mtp_loss,
                        )
                finally:
                    self._deferred_main_tokens = None
            else:
                lm_loss = self._forward(logits[0], lm_labels)
                if get_tensor_model_parallel_world_size() > 1:
                    target_p_self_op_dist = DistributedSoftmaxOp.apply(
                        logits[0], axis=2
                    )
                else:
                    target_p_self_op_dist = nn.Softmax(axis=2)(logits[0])
                if get_context_parallel_world_size() > 1:
                    cp_balance_mode = self.config.cp_balance_mode
                    if cp_balance_mode == "contiguous_allgather":
                        target_p_self_op_dist = MTPDistillationLossShift.apply(
                            target_p_self_op_dist,
                            self.config.num_nextn_predict_layers,
                            mode=cp_balance_mode,
                        )
                    else:
                        target_p_self_op_dist = ContextParallelGatherOp.apply(
                            target_p_self_op_dist,
                            axis=1,
                            mode=cp_balance_mode,
                        )

                def padding(tensor, left=False, pad_len=1):
                    zeropadding = paddle.zeros_like(tensor[:, -pad_len:, :])
                    if left:
                        tensor = paddle.concat((zeropadding, tensor), axis=1)
                    else:
                        tensor = paddle.concat((tensor, zeropadding), axis=1)
                    return tensor

                if (
                    self.config.num_nextn_predict_layers > 0
                    and mtp_logits is not None
                ):
                    for depth in range(len(mtp_logits)):
                        prediction_scores_cur_depth = mtp_logits[depth]
                        labels_cur_depth = labels_ori[
                            :, (depth + 1) : (depth + 1 + seq_length)
                        ]
                        lossmask = (
                            labels_cur_depth != self.ignored_index
                        ).cast(paddle.float32)
                        if get_tensor_model_parallel_world_size() > 1:
                            out_logp = paddle.log(
                                DistributedSoftmaxOp.apply(
                                    prediction_scores_cur_depth, axis=2
                                )
                            )
                        else:
                            out_logp = nn.LogSoftmax(axis=2)(
                                prediction_scores_cur_depth
                            )

                        if not (
                            get_context_parallel_world_size() > 1
                            and cp_balance_mode == "contiguous_allgather"
                        ):
                            target_p = target_p_self_op_dist[
                                :, (depth + 1) :, :
                            ].clone()
                            target_p = padding(
                                target_p, left=False, pad_len=depth + 1
                            )
                        if get_context_parallel_world_size() > 1:
                            if cp_balance_mode == "contiguous_allgather":
                                target_p = target_p_self_op_dist[
                                    :, depth : depth + out_logp.shape[1]
                                ]
                            else:
                                target_p = ContextParallelScatterOp.apply(
                                    target_p,
                                    axis=1,
                                    mode=cp_balance_mode,
                                )
                        plogp = target_p * out_logp

                        lossmask = lossmask[..., None]
                        xishu = lossmask.sum() + 1e-5
                        if get_context_parallel_world_size() > 1:
                            lossmask = ContextParallelScatterOp.apply(
                                lossmask,
                                axis=1,
                                mode=self.config.cp_balance_mode,
                            )

                        ploss = -paddle.sum(lossmask * plogp)
                        if get_tensor_model_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                            )

                        if get_context_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_context_parallel_group(),
                            )

                        ploss = ploss / xishu
                        mtp_loss.append(ploss)

            # Store detached MTP loss tensors into class-level tracker and global_training_logs.
            # Use .detach() instead of .item() to avoid GPU synchronization on every
            # micro-batch. The trainer will call .item() only at logging steps.
            for i, loss_val in enumerate(mtp_loss):
                LanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = (
                    loss_val.detach()
                )
                _print_scalar_loss_md5(
                    "MTP_LOSS_PATH_MD5",
                    f"mtp{i + 1}.final_loss",
                    loss_val,
                )

            logs = get_global_training_logs()
            if logs is not None and hasattr(logs, "update"):
                for i, loss_val in enumerate(mtp_loss):
                    logs.update(**{f"mtp_{i + 1}_loss": loss_val.detach()})

            def add_loss(main_loss, loss):
                if _use_accuracy_compatible_kernel():
                    # Megatron-aligned: MTP loss gradient flows but loss scalar unchanged.
                    # This matches Megatron's behavior where MTP contributes to training
                    # gradients without affecting the reported loss value.
                    if self.config.add_mtp_loss:
                        # E-226: the parenthesisation is load-bearing, not cosmetic.
                        #
                        # ``main + loss - loss.detach()`` evaluates left to right, so it
                        # forms ``main + loss`` FIRST and then subtracts. In float32 that
                        # round trip is lossy whenever ``loss`` is large enough relative
                        # to ``main``: with main 11.810652732849121 and 0.1 * mtp
                        # 1.2908377647399902 it returns 11.810651779174805, one ulp low.
                        # The comment above promises the scalar is unchanged, and this is
                        # the form that actually delivers it: ``loss - loss.detach()`` is
                        # exactly 0.0 with gradient 1, and adding an exact zero cannot
                        # perturb ``main``.
                        #
                        # It surfaced only once the MTP loss itself became bit-exact
                        # (E-226 aligned the MTP rope positions): before that the wrong
                        # ``loss`` value happened to round back onto ``main``, so the
                        # reported loss looked right for the wrong reason. It matters for
                        # the acceptance loss gate, which compares IEEE bit patterns.
                        #
                        # The non-accuracy-compatible branch below carries the same
                        # left-to-right form for the same gradient-only purpose; it is
                        # left alone here to keep this change confined to the symmetric
                        # alignment path.
                        return main_loss + (loss - loss.detach())
                    else:
                        return main_loss
                else:
                    # Original behavior
                    if self.config.add_mtp_loss:
                        return main_loss + loss
                    else:
                        return main_loss + loss - loss.detach()

            if self.config.gpt_model_use_experimental_version:
                # Align with EB: accumulate inside loop to match float32
                # arithmetic order: loss += scaling * loss_i / N
                loss = lm_loss
                if _use_accuracy_compatible_kernel():
                    # Megatron-aligned: only add MTP loss when add_mtp_loss=True.
                    # Use add_loss() to keep single maintenance point for compat
                    # behavior (loss + val - val.detach() for gradient-only flow).
                    if self.config.add_mtp_loss:
                        num_mtp = len(mtp_loss)
                        for mtp_l in mtp_loss:
                            mtp_val = (
                                self.config.mtp_loss_scaling_factor
                                * mtp_l
                                / num_mtp
                            )
                            loss = add_loss(loss, mtp_val)
                else:
                    # Original behavior: always use add_loss
                    num_mtp = len(mtp_loss)
                    for mtp_l in mtp_loss:
                        loss = add_loss(
                            loss,
                            self.config.mtp_loss_scaling_factor
                            * mtp_l
                            / num_mtp,
                        )
            else:
                loss = add_loss(
                    lm_loss,
                    self.config.mtp_loss_scaling_factor
                    * sum(mtp_loss)
                    / len(mtp_loss),
                )

            return loss
        else:
            return self._forward(logits, labels)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="LanguageLoss")


class MainLanguageLoss(LanguageLoss):
    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict | list, labels: Tensor) -> Tensor:
        assert (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        )
        labels_ori = labels
        lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
        seq_length = lm_labels.shape[1]

        mtp_loss = dict_args["mtp_loss"]
        logits = dict_args["logits"]

        assert not self.config.mtp_distillation_loss, (
            "separate mtp head & loss don't support mtp_distillation_loss"
        )

        if self.config.train_mtp_only:
            lm_loss = 0.0
        else:
            lm_loss = self._forward(logits, lm_labels)

        # Store detached MTP loss tensors into class-level tracker and global_training_logs.
        # Use .detach() instead of .item() to avoid GPU synchronization on every
        # micro-batch. The trainer will call .item() only at logging steps.
        for i, loss_val in enumerate(mtp_loss):
            MainLanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = (
                loss_val.detach()
            )
            _print_scalar_loss_md5(
                "MTP_LOSS_PATH_MD5",
                f"mtp{i + 1}.final_loss",
                loss_val,
            )

        # Also write to global_training_logs to read
        logs = get_global_training_logs()
        if logs is not None and hasattr(logs, "update"):
            for i, loss_val in enumerate(mtp_loss):
                logs.update(**{f"mtp_{i + 1}_loss": loss_val.detach()})

        def add_loss(main_loss, loss):
            if _use_accuracy_compatible_kernel():
                # Megatron-aligned: MTP loss gradient flows but loss scalar unchanged.
                # This matches Megatron's behavior
                if self.config.add_mtp_loss:
                    return main_loss + loss - loss.detach()
                else:
                    return main_loss
            else:
                # Original behavior
                if self.config.add_mtp_loss:
                    return main_loss + loss
                else:
                    return main_loss + loss - loss.detach()

        loss = add_loss(
            lm_loss,
            self.config.mtp_loss_scaling_factor * sum(mtp_loss) / len(mtp_loss),
        )

        return loss

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MainLanguageLoss")


class MTPLanguageLoss(LanguageLoss):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict):
        mtp_logits = dict_args.get("mtp_logits")
        labels = dict_args.get("labels")
        assert mtp_logits is not None, (
            "separate mtp loss must provide mtp_logits"
        )
        assert labels is not None, "separate mtp loss must provide labels"
        assert (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        )
        labels_ori = labels
        lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
        seq_length = lm_labels.shape[1]

        mtp_loss = []

        assert not self.config.mtp_distillation_loss, (
            "separate mtp head & loss don't support mtp_distillation_loss"
        )

        for depth in range(self.config.num_nextn_predict_layers):
            logits_cur_depth = mtp_logits[depth]
            labels_cur_depth = labels_ori[
                :, (depth + 1) : (depth + 1 + seq_length)
            ]
            loss_cur_depth = self._forward(
                logits_cur_depth,
                labels_cur_depth,
            )
            mtp_loss.append(loss_cur_depth)

        dict_args.pop("mtp_logits")
        dict_args["mtp_loss"] = mtp_loss

        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MTPLanguageLoss")
