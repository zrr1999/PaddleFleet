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

from __future__ import annotations

import functools
import hashlib
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
import paddlefleet_ops
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

if TYPE_CHECKING:
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.transformer_config import TransformerConfig

from paddlefleet import utils
from paddlefleet.recompute_utils import need_recompute_in_first_n
from paddlefleet.transformer.activations import situ
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.utils import profile

from .fp8_utils import fused_stack_quant_without_cache
from .fused_a2a import configure_buffer
from .fusion_layer_utils import (
    FusionMoePyLayer,
    HybridEPMoePyLayer,
)
from .moe_expert import GroupedMLPExpert, SonicMoEExpert, StandardMLPExpert
from .moe_router import TopKRouter
from .moe_shared_expert import StandardMLPSharedExpert
from .moe_utils import AddAuxiliaryLoss, use_accuracy_compatible_kernel
from .token_dispatcher import (
    AllGatherTokenDispatcher,
    AllToAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    is_hybrid_ep_backend_selected,
)

logger = logging.getLogger(__name__)


# MD5 logging for MoE precision debugging
_LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"


def _log_moe_md5(tensor, name, layer_idx=None):
    """Log MD5 of a tensor for MoE precision alignment debugging."""
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    if _LOG_LAYER_MD5 and TransformerLayer._gpt_model_use_experimental_version:
        if TransformerLayer._skip_mtp_probes:
            return  # Skip MTP passes — EC has no MTP
        data = tensor.detach().cast("float32").numpy().tobytes()
        md5 = hashlib.md5(data).hexdigest()
        rank = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )
        layer_str = f" Layer={layer_idx}" if layer_idx is not None else ""
        print(
            f"[MD5 MoE] Rank={rank}{layer_str} {name} MD5={md5} shape={list(tensor.shape)}",
            flush=True,
        )


from .moe_utils import (
    global_moe_balance_training_logs_enabled,
    log_moe_balance,
    log_moe_losses,
    permute,
    unpermute,
)


class GradDtypeGuard(PyLayer):
    """Guard the grad's dtype if different from input's dtype."""

    @staticmethod
    def forward(ctx, x, dtype):
        """forward"""
        return paddle.empty([0], dtype=dtype), {"x": x}

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return grad


class GradDtypeUnguard(PyLayer):
    """Remove grad dtype guard."""

    @staticmethod
    def forward(ctx, x, status):
        """forward"""
        if hasattr(ctx, "set_grad_in_dtype_consistent"):
            ctx.set_grad_in_dtype_consistent(False)
        return status["x"]

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return grad


class ThreePathCloneAlignMG(PyLayer):
    """Three-way differentiable identity clone with MG-aligned backward sum order."""

    @staticmethod
    def forward(ctx, x):
        return x.clone(), x.clone(), x.clone()

    @staticmethod
    def backward(ctx, g_router, g_dispatcher, g_shared):
        partial = g_dispatcher + g_shared
        out = partial + g_router
        return out


@dataclass
class MoESublayers:
    """MoE Layer Sublayers spec"""

    mlp_spec: LayerSpec | type = None  # Used by experts


class MoELayer(nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MoESublayers | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        self.config = config
        self.use_accuracy_compatible = getattr(
            config, "use_accuracy_compatible", False
        )
        self.moe_sublayers = sublayers
        routed_expert_config = deepcopy(config)
        shared_expert_config = deepcopy(config)
        global_use_bias = routed_expert_config.use_bias
        moe_routed_expert_use_bias = config.moe_routed_expert_use_bias
        if moe_routed_expert_use_bias is not None:
            routed_expert_config.use_bias = moe_routed_expert_use_bias
            logger.info(
                "PaddleFleet MoELayer moe_routed_expert_use_bias overrides "
                "routed_expert_config.use_bias: global_use_bias=%s moe_routed_expert_use_bias=%s",
                global_use_bias,
                moe_routed_expert_use_bias,
            )
        self.pg_collection = pg_collection
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.moe_shared_expert_intermediate_size = None
        if self.n_shared_experts:
            self.moe_shared_expert_intermediate_size = (
                self.moe_intermediate_size * self.n_shared_experts
            )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_act = config.hidden_act
        self.sequence_parallel = config.sequence_parallel
        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type
        self.moe_allgather_gate_overlap = config.moe_allgather_gate_overlap
        self.use_hybrid_ep_backend = False
        self.moe_shared_expert_overlap = config.moe_shared_expert_overlap
        self.fp8 = config.fp8
        self.use_ue8m0 = config.use_ue8m0
        self.use_w4a8 = config.use_w4a8
        self.use_w4a8_fused_quant = config.use_w4a8_fused_quant
        self.dw_p2p_overlap = getattr(config, "dw_p2p_overlap", False)
        self.using_sonic_moe = self.config.using_sonic_moe
        self.fp8_dispatch = bool(config.fp8) and not self.use_w4a8
        self.fp8_wgrad = config.fp8_wgrad
        self.fp8_dispatch_bwd = (
            self.fp8_dispatch and self.using_sonic_moe and self.fp8_wgrad
        )
        self.moe_expert_fusion = config.moe_expert_fusion
        import os as _os

        if _os.environ.get("MODEL_REPRO_MOE_FUSION", "0") == "1":
            print(
                f"[MOE-FUSION-DEBUG] MoELayer init moe_expert_fusion={self.moe_expert_fusion} "
                f"ep={getattr(self, 'expert_model_parallel_size', None)} "
                f"dispatch={getattr(self, 'moe_token_dispatcher_type', None)}",
                flush=True,
            )
        if self.hidden_act == situ and (
            config.moe_use_fusion_node or self.moe_expert_fusion
        ):
            raise ValueError(
                "SiTU-GLU does not support moe_use_fusion_node=True or "
                "moe_expert_fusion=True yet; support will be added in a future "
                "release. Please set both options to False."
            )
        self.moe_subbatch_token_num_after_dispatch = (
            config.moe_subbatch_token_num_after_dispatch
        )
        if self.using_sonic_moe:
            assert paddlefleet_ops.is_sonic_moe_available(), (
                paddlefleet_ops.blocked_import_messages[
                    "paddlefleet_ops.sonicmoe"
                ]
            )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.moe_deep_gemm = config.moe_deep_gemm

        if self.moe_deep_gemm:
            incompatible_reasons = []
            if not self.moe_expert_fusion:
                incompatible_reasons.append("moe_expert_fusion must be True")
            if incompatible_reasons:
                logging.warning(
                    "moe_deep_gemm=True is ignored because %s; "
                    "setting moe_deep_gemm to False.",
                    " and ".join(incompatible_reasons),
                )
                self.moe_deep_gemm = False
        self.moe_ep_barrier = config.moe_ep_barrier

        # Latent MoE initialization
        self.use_latent_moe = (
            self.config.moe_latent_size is not None
            and self.config.moe_latent_size > 0
        )
        if self.use_latent_moe:
            logging.info(
                f"Latent MoE enabled: hidden_size={self.config.hidden_size} -> moe_latent_size={self.config.moe_latent_size}"
            )
            self.fc1_latent_proj = nn.Linear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                bias_attr=self.config.use_bias,
            )
            self.fc2_latent_proj = nn.Linear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                bias_attr=self.config.use_bias,
            )
            # Override default XavierUniform with config init methods
            self.config.init_method(self.fc1_latent_proj.weight)
            self.config.output_layer_init_method(self.fc2_latent_proj.weight)
            self.latent_norm = (
                WrappedPaddleNorm(
                    config=self.config,
                    hidden_size=self.config.moe_latent_size,
                    eps=self.config.rms_norm_eps,
                )
                if self.config.latent_moe_use_norm
                else None
            )
            # Update expert config to use latent size
            routed_expert_config.hidden_size = self.config.moe_latent_size
        # Cached latent-space projection from _maybe_pre_allgather_overlap;
        # consumed (and cleared) by _project_to_latent. Initialised here so the
        # attribute always exists regardless of which forward entry path is
        # taken (custom_forward vs fusion_moe_forward) and whether overlap fired.
        self._latent_hidden = None
        self.moe_group = pg_collection.ep
        self.expert_model_parallel_size = (
            utils.get_pg_size(self.moe_group)
            if self.moe_group is not None
            else 1
        )
        self.num_local_experts = (
            self.num_experts // self.expert_model_parallel_size
        )
        # MoE-Related Configs
        self._init_expert_parallel()

        self.gate = TopKRouter(config=config, pg_collection=pg_collection)

        self.expert_class = StandardMLPExpert
        self.shared_expert_class = StandardMLPSharedExpert

        if (
            self.expert_model_parallel_size <= 1
            and self.sequence_parallel
            and self.tensor_model_parallel_size > 1
        ):
            routed_expert_config.sequence_parallel = False
            if not self.config.gpt_model_use_experimental_version:
                shared_expert_config.sequence_parallel = False
        elif (
            self.expert_model_parallel_size > 1
            and self.tensor_model_parallel_size >= 1
            or paddle.version.cuda() == "12.6"
        ):
            routed_expert_config.tensor_model_parallel_size = 1

        if (
            paddle.is_compiled_with_cuda()
            and paddle.device.get_device_capability()[0] < 9
        ):
            # TODO: Support Ampere architecture after upgrade deepep in paddlepaddle
            if self.moe_token_dispatcher_type in ("deepep", "hybridep"):
                logger.info(
                    "deepep/hybridep in paddlepaddle does not support compute capability < 9.0, "
                    "fallback to alltoall token dispatcher."
                )
                self.moe_token_dispatcher_type = "alltoall"
            if self.moe_deep_gemm:
                logger.warning(
                    "moe_deep_gemm is not supported when device capability < 9.0."
                )
                self.moe_deep_gemm = False

        self.moe_use_fusion_node = config.moe_use_fusion_node
        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type in ("deepep", "hybridep"):
                self.use_hybrid_ep_backend = is_hybrid_ep_backend_selected(
                    self.moe_token_dispatcher_type
                )
                if (
                    self.moe_use_fusion_node
                    and self.use_hybrid_ep_backend
                    and self.moe_shared_expert_overlap
                ):
                    logger.info(
                        "HybridEP backend does not support moe_shared_expert_overlap; disabling it."
                    )
                    self.moe_shared_expert_overlap = False
            elif self.moe_token_dispatcher_type == "allgather":
                self._validate_allgather_config()
            else:
                logger.info(
                    "moe_use_fusion_node is only supported when moe_token_dispatcher_type is 'deepep' or 'hybridep'; disabling it."
                )
                self.moe_use_fusion_node = False
                if self.moe_expert_fusion:
                    raise ValueError(
                        "moe_expert_fusion is only supported when moe_token_dispatcher_type is 'deepep' or 'hybridep' and on GPU architecture SM90 or higher. If these conditions are not met, please set it to false in the configuration yaml."
                    )
                self.fp8_dispatch = False

        if self.fp8:
            if paddle.version.cuda() == "12.6":
                raise NotImplementedError(
                    "fp8 is not supported when cuda version == 12.6."
                )
            assert self.moe_use_fusion_node, (
                "fp8 can only be used when moe_use_fusion_node = True."
            )

        if self.use_ue8m0:
            assert paddle.device.cuda.get_device_capability()[0] == 10, (
                "use_ue8m0 requires Blackwell GPU (SM100)"
            )

        expert_args = {}
        expert_args["config"] = routed_expert_config
        expert_args["moe_intermediate_size"] = self.moe_intermediate_size
        expert_args["is_expert"] = True
        expert_args["mlp_spec"] = self.moe_sublayers.mlp_spec

        use_fused_weight = self.moe_expert_fusion
        if (
            self.fp8
            and (self.moe_expert_fusion is False)
            and self.moe_deep_gemm
        ):
            raise ValueError(
                "For fp8 deep_gemm (i.e. use k-grouped gemm in backward), moe_expert_fusion must be True."
            )
        if (
            self.fp8
            and self.moe_expert_fusion
            and self.moe_deep_gemm is False
            and self.using_sonic_moe is False
        ):
            use_fused_weight = False
        if self.using_sonic_moe:
            assert use_fused_weight is True, (
                "for sonic moe, expert weight must be fused."
            )

        if self.fp8 and self.using_sonic_moe is False:
            logger.warning(
                f"fp8_weight_quant_format ({self.config.fp8_weight_quant_format}) configuration currently only works in SonicMoE."
            )

        if use_fused_weight:
            if (
                self.moe_token_dispatcher_type == "allgather"
                and self.expert_model_parallel_size > 1
            ):
                # AllGather EP>1: every rank holds all experts, sharded
                # along intermediate dim (I // EP per rank).
                self.grouped_gemm_experts = SonicMoEExpert(
                    self.num_experts,
                    self.num_experts_per_tok,
                    routed_expert_config,
                    pg_collection,
                    intermediate_size_per_partition=(
                        self.moe_intermediate_size
                        // self.expert_model_parallel_size
                    ),
                )
            elif self.using_sonic_moe:
                # TODO: replace grouped_gemm_experts with fusion_experts
                self.grouped_gemm_experts = SonicMoEExpert(
                    self.num_local_experts,
                    self.num_experts_per_tok,
                    routed_expert_config,
                    pg_collection,
                )
            else:
                # TODO: replace grouped_gemm_experts with fusion_experts
                self.grouped_gemm_experts = GroupedMLPExpert(
                    self.num_local_experts,
                    routed_expert_config,
                    self.moe_deep_gemm,
                    pg_collection,
                )
        else:
            self.experts = nn.LayerList([])
            for i in range(self.num_experts):
                if i // self.num_experts_per_device == self.moe_rank:
                    self.experts.append(self.expert_class(**expert_args))
                else:
                    self.experts.append(None)

        shared_expert_args = deepcopy(expert_args)
        if self.config.gpt_model_use_experimental_version:
            shared_expert_args["is_expert"] = False
            shared_expert_args["config"] = shared_expert_config
        shared_expert_args["config"].use_bias = shared_expert_config.use_bias
        shared_expert_args["config"].hidden_size = self.config.hidden_size
        shared_expert_args["moe_intermediate_size"] = (
            self.moe_shared_expert_intermediate_size
        )
        shared_expert_args["is_expert"] = False
        if (
            os.environ.get("MODEL_REPRO_MOE_SHARED_TP", "0") == "1"
            and self.pg_collection is not None
        ):
            # E-127: mcore's SharedExpertMLP is a TP-split MLP (linear_fc1 column
            # parallel over TP, linear_fc2 row parallel + allreduce), while
            # PaddleFleet's default shared expert keeps full per-rank weights.
            # The bf16 GEMM over the full N vs half-N+reduce differ by 1 ULP per
            # output after downstream combine (~8k/184320 at layer3). Restore the
            # TP-split structure (weights load through the same TP splitter as
            # dense MLPs).
            shared_expert_args["tp_group"] = self.pg_collection.tp
        if self.n_shared_experts > 0:
            self.shared_experts = self.shared_expert_class(**shared_expert_args)
            try:
                self.shared_experts._shared_layer_no = str(getattr(self, "layer_number", "-"))
            except Exception:
                pass
        else:
            self.shared_experts = None

        # when sp is enabled, mark shared_experts as sequence parallel, because:
        # 1. shared_experts only process local tokens which shape is [s/tp,b,h]
        # 2. shared_experts'weight and bias will not be splited across tp ranks
        if (
            not self.config.gpt_model_use_experimental_version
            and self.sequence_parallel
            and self.expert_model_parallel_size > 1
            and self.shared_experts is not None
        ):
            mark_as_sequence_parallel_parameter(
                self.shared_experts.up_gate_proj.weight
            )
            if shared_expert_config.use_bias:
                mark_as_sequence_parallel_parameter(
                    self.shared_experts.up_gate_proj.bias
                )
            mark_as_sequence_parallel_parameter(
                self.shared_experts.down_proj.weight
            )
            if shared_expert_config.use_bias:
                mark_as_sequence_parallel_parameter(
                    self.shared_experts.down_proj.bias
                )

        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type in ("deepep", "hybridep"):
                # Set NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN automatically if not set by user.
                if (
                    self.moe_token_dispatcher_type == "hybridep"
                    and os.getenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
                    is None
                ):
                    # We limit the default domain size to 64 due to NVL72 topology. If user wants to use
                    # a larger domain size, they can set the environment variable manually.
                    num_of_hybrid_ep_ranks_per_nvlink_domain = min(
                        self.expert_model_parallel_size, 64
                    )
                    os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = (
                        str(num_of_hybrid_ep_ranks_per_nvlink_domain)
                    )
                    logger.info(
                        "Automatically set NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=%d for hybrid EP backend.",
                        num_of_hybrid_ep_ranks_per_nvlink_domain,
                    )
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_experts_per_device,
                    self.num_experts_per_tok,
                    self.num_experts,
                    self.moe_group,
                    self.moe_ep_barrier,
                    dispatcher_type=self.moe_token_dispatcher_type,
                    hybridep_buffer_configs=getattr(
                        config, "hybridep_buffer_configs", None
                    ),
                    moe_deep_gemm=self.moe_deep_gemm,
                    use_accuracy_compatible=getattr(
                        self, "use_accuracy_compatible", False
                    ),
                )
                if (
                    self.moe_token_dispatcher_type == "deepep"
                    and getattr(config, "deepep_buffer_configs", None)
                    is not None
                ):
                    configure_buffer(**config.deepep_buffer_configs)
            elif self.moe_token_dispatcher_type == "alltoall":
                local_expert_indices = list(
                    range(
                        self.moe_rank * self.num_experts_per_device,
                        (self.moe_rank + 1) * self.num_experts_per_device,
                    )
                )
                self.token_dispatcher = AllToAllTokenDispatcher(
                    self.moe_group,
                    self.expert_model_parallel_size,
                    self.num_experts_per_device,
                    local_expert_indices,
                    use_accuracy_compatible=getattr(
                        self, "use_accuracy_compatible", False
                    ),
                )
            elif self.moe_token_dispatcher_type == "allgather":
                self.token_dispatcher = AllGatherTokenDispatcher(
                    self.moe_group,
                    self.expert_model_parallel_size,
                    self.num_experts,
                    fp8_dispatch=self.fp8_dispatch,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported moe_token_dispatcher_type {self.moe_token_dispatcher_type}"
                )

        self.recompute_moe_gate_up = getattr(
            self.config, "recompute_moe_gate_up", False
        ) or (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "moe_gate_up" in self.config.recompute_modules
        )
        self.recompute_moe_premute = getattr(
            self.config, "recompute_moe_premute", False
        ) or (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "moe_premute" in self.config.recompute_modules
        )
        self.use_auto_subbatch = getattr(
            self.config, "use_auto_subbatch", False
        )
        self.moe_subbatch_diag = getattr(
            self.config, "moe_subbatch_diag", False
        )
        self.auto_subbatch_mode = getattr(
            self.config, "auto_subbatch_mode", None
        )

        if self.expert_model_parallel_size > 1:
            self.is_mp_moe = False
            self.is_ep_moe = True
            # Color routed-expert params with the default "moe_expert" now,
            # matching the historical construction-time behavior, UNLESS
            # mtp_shared_last_layer is enabled. In the shared-MTP case the color
            # (moe_expert vs the no-hook variant) depends on the layer number,
            # which is unknown here, so coloring is deferred to
            # set_layer_number()/_color_expert_params(). Paddle forbids
            # reassigning a non-None color, so coloring must happen exactly once.
            color_experts_now = not getattr(
                self.config, "mtp_shared_last_layer", False
            )
            fusion_experts = None
            if hasattr(self, "grouped_gemm_experts"):
                fusion_experts = self.grouped_gemm_experts
            if fusion_experts is not None:
                for p in fusion_experts.parameters():
                    p.is_moe_param = True
                    # Default color set here; deferred when mtp_shared_last_layer
                    # is on (see set_layer_number/_color_expert_params).
                    if color_experts_now:
                        p.color = {
                            "color": "moe_expert",
                            "group": self.moe_grad_group,
                        }
                    p.no_sync = not self.is_mp_moe
                    p.expert = not self.is_mp_moe
                    if self.is_mp_moe or self.is_ep_moe:
                        p.is_distributed = True
            else:
                assert self.experts is not None, (
                    "experts should be initialized."
                )
                for p in self.experts.parameters():
                    p.is_moe_param = True
                    # Default color set here; deferred when mtp_shared_last_layer
                    # is on (see set_layer_number/_color_expert_params).
                    if color_experts_now:
                        p.color = {
                            "color": "moe_expert",
                            "group": self.moe_grad_group,
                        }
                    p.no_sync = not self.is_mp_moe
                    p.expert = not self.is_mp_moe
                    if self.is_mp_moe or self.is_ep_moe:
                        p.is_distributed = True

        self.use_rr_deepep_combine = False

    def rr_recompute_update(self, in_full_recompute, in_mlp_recompute):
        if (
            self.config.recompute_modules is not None
            and "moe_combine" in self.config.recompute_modules
        ):
            if (
                self.moe_token_dispatcher_type != "deepep"
                or not self.moe_shared_expert_overlap
            ):
                raise ValueError(
                    "moe_combine RR is only supported in DeepEP mode with "
                    "moe_shared_expert_overlap enabled (combine_overlap scenario)."
                )
            if self.config.recompute_granularity is None:
                raise ValueError(
                    "recompute_granularity must be set when moe_combine RR is enabled."
                )
            if isinstance(self.config.recompute_modules, list):
                self.use_rr_deepep_combine = True
            elif isinstance(self.config.recompute_modules, dict):
                # dict mode only supports first_n: uniform applies recompute to all layers
                # (use list mode instead), block is not yet implemented but can be extended.
                if self.config.recompute_method != "first_n":
                    raise ValueError(
                        "recompute_modules dict mode for moe_combine RR requires "
                        f"recompute_method='first_n', got '{self.config.recompute_method}'."
                    )
                if not hasattr(self, "layer_number"):
                    raise ValueError(
                        "layer_number must be set before rr_recompute_update is called in dict mode. "
                        "Ensure set_layer_number() is called first."
                    )
                self.use_rr_deepep_combine = not need_recompute_in_first_n(
                    self.layer_number,
                    self.config,
                    self.config.recompute_modules["moe_combine"],
                )
        if (
            (not in_full_recompute)
            and (not in_mlp_recompute)
            and self.use_rr_deepep_combine
        ):
            raise ValueError(
                "Enabling rr for moe_combine is meaningless when neither full_recompute "
                "nor mlp_recompute is active."
            )

    def _init_expert_parallel(self):
        def _parse_moe_expert_parallel(
            num_experts: int, expert_model_parallel_size: int
        ) -> int:
            """
            Args:
                num_experts: Total number of experts
                expert_model_parallel_size: Expert parallel groups

            Returns:
                n_routed_experts_per_device: Number of experts per device
            """
            assert num_experts >= expert_model_parallel_size, (
                f"expert num_experts={num_experts} >= moe_world_size={expert_model_parallel_size}"
            )
            assert num_experts % expert_model_parallel_size == 0, (
                f"expert num_experts={num_experts} % moe_world_size={expert_model_parallel_size} == 0"
            )

            n_routed_experts_per_device = (
                num_experts // expert_model_parallel_size
            )
            return n_routed_experts_per_device

        if self.expert_model_parallel_size > 1:
            self.moe_grad_group = self.pg_collection.expt_dp
            self.moe_rank = utils.get_pg_rank(self.moe_group)
            self.moe_rank = max(self.moe_rank, 0)
            if self.moe_token_dispatcher_type == "allgather":
                # AllGather: every rank holds a shard of every expert.
                self.num_experts_per_device = self.num_experts
            else:
                self.num_experts_per_device = _parse_moe_expert_parallel(
                    self.num_experts, self.expert_model_parallel_size
                )
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_model_parallel_size = 1
            self.num_experts_per_device = self.num_experts

    def expert_forward(
        self,
        dispatched_input,
        tokens_per_expert,
    ):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist()
            if not isinstance(tokens_per_expert, list)
            else tokens_per_expert
        )
        chunks = paddle.split(
            dispatched_input, num_or_sections=tokens_per_expert, axis=0
        )
        scale_chunks = None
        if use_accuracy_compatible_kernel():
            per_token_scale = getattr(
                self.token_dispatcher, "global_input_probs", None
            )
            if per_token_scale is None:
                raise RuntimeError(
                    "FLAGS_use_accuracy_compatible_kernel requires dispatched "
                    "router probabilities from the token dispatcher."
                )
            scale_chunks = paddle.split(
                per_token_scale, num_or_sections=tokens_per_expert, axis=0
            )
        for i, chunk in enumerate(chunks):
            if tokens_per_expert[i] == 0:
                continue
            chunk = chunk.contiguous()
            current_expert_idx = i + self.moe_rank * self.num_experts_per_device
            expert = self.experts[current_expert_idx]
            if scale_chunks is None:
                expert_output = expert(chunk)[0]
            else:
                expert_output = expert(chunk, per_token_scale=scale_chunks[i])[
                    0
                ]
            outputs += [expert_output]

        if not outputs:
            return dispatched_input

        return paddle.concat(outputs, axis=0)

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
        async_finish: bool = False,
    ):
        hidden_states = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, routing_map, topk_weights, topk_indices
        )
        hidden_states, fp8_dispatched_handle = (
            self.token_dispatcher.token_dispatch(
                hidden_states,
                self.fp8_dispatch,
                async_finish=async_finish,
                use_ue8m0=self.use_ue8m0,
                using_sonic_moe=self.using_sonic_moe,
            )
        )
        return hidden_states, fp8_dispatched_handle

    def permute(self, hidden_states: paddle.Tensor):
        global_input_tokens, tokens_per_expert = (
            self.token_dispatcher.dispatch_postprocess(hidden_states)
        )
        return global_input_tokens, tokens_per_expert

    def unpermute(self, hidden_states: paddle.Tensor):
        return self.token_dispatcher.combine_preprocess(hidden_states)

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        """Combine expert outputs back to the local token shard.

        For the 'allgather' and 'alltoall' dispatchers: ``token_combine``
        issues the reverse communication, then ``combine_postprocess``
        finalizes it.

        For other dispatchers (deepep / hybridep): delegates to
        ``_comm_manager.combine`` directly, which already returns the restored
        tensor (no separate combine_postprocess step).
        """
        if self.moe_token_dispatcher_type in ("allgather", "alltoall"):
            hidden_states = self.token_dispatcher.token_combine(
                hidden_states,
                combine_overlap_handle=combine_overlap_handle,
                async_finish=async_finish,
                fp8_combine_grad_handle=fp8_combine_grad_handle,
            )
            return self.token_dispatcher.combine_postprocess(hidden_states)
        return self.token_dispatcher._comm_manager.combine(
            hidden_states,
            combine_overlap_handle,
            use_rr_deepep_combine=self.use_rr_deepep_combine,
            fp8_dispatch=self.fp8_dispatch_bwd,
            combine_grad_handle=fp8_combine_grad_handle,
        )

    def routed_experts_compute(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens, tokens_per_expert = self.permute(hidden_states)
        expert_outs = self.expert_forward(
            global_input_tokens,
            tokens_per_expert,
        )
        return self.unpermute(expert_outs)

    def _maybe_pre_allgather_overlap(self, hidden_states: paddle.Tensor):
        """Pre-issue async AllGather on comm stream to overlap with gate MLP.

        allgather + EP>1 + moe_allgather_gate_overlap only. Result consumed
        in dispatch_preprocess. For latent MoE, fc1_latent_proj is hoisted
        here so AllGather targets latent-space tensor.
        """
        if (
            self.moe_token_dispatcher_type == "allgather"
            and self.expert_model_parallel_size > 1
            and self.moe_allgather_gate_overlap
        ):
            if self.use_latent_moe:
                self._latent_hidden = self.fc1_latent_proj(hidden_states)
                self.token_dispatcher.pre_allgather(self._latent_hidden)
            else:
                self._latent_hidden = None
                self.token_dispatcher.pre_allgather(hidden_states)
        else:
            self._latent_hidden = None

    def _validate_allgather_config(self):
        """Validate and force-correct config flags for the allgather dispatcher.

        AllGather + ReduceScatter EP pattern: every expert is sharded along its
        intermediate dim across the EP group.  Requires SonicMoE fused kernels;
        fp8 dispatch quantization is handled by ``AllGatherTokenDispatcher``
        (see ``_quantize_and_pack_fp8``) and fp8 expert compute by
        ``run_sonic_moe``.
        """
        if not self.using_sonic_moe:
            raise ValueError(
                "moe_token_dispatcher_type='allgather' requires "
                "using_sonic_moe=True; the allgather path is only "
                "implemented for SonicMoE fused kernels."
            )
        if not self.moe_use_fusion_node:
            logger.warning(
                "moe_token_dispatcher_type='allgather' only "
                "support moe_use_fusion_node; forcing moe_use_fusion_node=True."
            )
            self.moe_use_fusion_node = True
        if not self.moe_expert_fusion:
            logger.warning(
                "moe_token_dispatcher_type='allgather' requires "
                "fused expert weights; forcing moe_expert_fusion=True."
            )
            self.moe_expert_fusion = True
        if self.moe_deep_gemm:
            logger.warning(
                "moe_token_dispatcher_type='allgather' does not "
                "support moe_deep_gemm; forcing moe_deep_gemm=False."
            )
            self.moe_deep_gemm = False
        if self.moe_intermediate_size % self.expert_model_parallel_size != 0:
            raise ValueError(
                f"moe_intermediate_size={self.moe_intermediate_size} "
                f"must be divisible by EP="
                f"{self.expert_model_parallel_size} in 'allgather' mode."
            )
        if self.fp8:
            intermediate_per_rank = (
                self.moe_intermediate_size // self.expert_model_parallel_size
            )
            if intermediate_per_rank % 128 != 0:
                raise ValueError(
                    f"allgather + fp8 requires "
                    f"moe_intermediate_size / EP to be divisible by 128 "
                    f"(fp8 block-scale tile), got "
                    f"moe_intermediate_size={self.moe_intermediate_size}, "
                    f"EP={self.expert_model_parallel_size}, "
                    f"intermediate_per_rank={intermediate_per_rank}. "
                    f"Consider reducing EP to a divisor of "
                    f"moe_intermediate_size // 128 = "
                    f"{self.moe_intermediate_size // 128}."
                )

    def _project_to_latent(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Project hidden_states to latent space, consuming any cached
        projection from the AllGather overlap path if available.
        """
        if not self.use_latent_moe:
            return hidden_states
        if self._latent_hidden is not None:
            hidden_states = self._latent_hidden
            self._latent_hidden = None
        else:
            hidden_states = self.fc1_latent_proj(hidden_states)
        return hidden_states

    # MoE forward: dispatch -> permute -> compute ->unpermute -> combine
    def custom_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        # Latent MoE: project hidden_states to latent space before dispatch
        if self.use_latent_moe:
            hidden_states = self.fc1_latent_proj(hidden_states)

        should_log_balance = framework._dygraph_tracer()._has_grad
        with profile("dispatch"):
            hidden_states, _ = self.dispatch(
                hidden_states, probs, routing_map, topk_weights, topk_indices
            )
        if should_log_balance and global_moe_balance_training_logs_enabled():
            log_moe_balance(
                self.layer_number,
                self.moe_group,
                self.num_experts_per_tok,
                self.token_dispatcher.get_dispatched_routing()[2],
                is_mtp_layer=self.is_mtp_layer,
            )
        with profile("fusion_mlp"):
            hidden_states = self.routed_experts_compute(hidden_states)
        with profile("combine"):
            hidden_states = self.combine(hidden_states)

        # Latent MoE: project back from latent space to hidden_size
        if self.use_latent_moe:
            if self.latent_norm is not None:
                hidden_states = self.latent_norm(hidden_states)
            hidden_states = self.fc2_latent_proj(hidden_states)

        return hidden_states

    def fusion_moe_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        combine_overlap_handle: dict,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        hidden_states = self._project_to_latent(hidden_states)

        should_log_balance = framework._dygraph_tracer()._has_grad
        with profile("dispatch"):
            dispatched_hidden_states, fp8_dispatched_handle = self.dispatch(
                hidden_states, probs, routing_map, topk_weights, topk_indices
            )

        dispatched_indices, dispatched_probs, tokens_per_expert = (
            self.token_dispatcher.get_dispatched_routing()
        )
        if should_log_balance and global_moe_balance_training_logs_enabled():
            log_moe_balance(
                self.layer_number,
                self.moe_group,
                self.num_experts_per_tok,
                tokens_per_expert,
                is_mtp_layer=self.is_mtp_layer,
            )
        fp8_combine_grad_handle = {} if self.fp8_dispatch_bwd else None
        # fp8_combine_grad_handle = None

        with profile("fusion_mlp"):
            if self._use_hybrid_ep_fusion():
                hidden_states = self._run_hybrid_ep_fusion(
                    dispatched_hidden_states,
                    dispatched_probs,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                )
            elif self.using_sonic_moe:
                use_fp8 = self.fp8 is not None
                fp8_scale = None
                if fp8_dispatched_handle is not None:
                    fp8_scale = fp8_dispatched_handle["scale"]
                hidden_states = self.grouped_gemm_experts(
                    dispatched_hidden_states,
                    dispatched_indices,
                    dispatched_probs,
                    use_fp8,
                    tokens_per_expert=tokens_per_expert,
                    fp8_scale=fp8_scale,
                    recompute_moe_gate_up=self.recompute_moe_gate_up,
                    fp8_combine_grad_handle=fp8_combine_grad_handle,
                )
            else:
                hidden_states = FusionMoePyLayer.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices,
                    self,
                    self.num_experts_per_tok,
                    use_fp8_mlp=self.fp8,
                    moe_deep_gemm=self.moe_deep_gemm,
                    recompute_moe_gate_up=self.recompute_moe_gate_up,
                    recompute_moe_premute=self.recompute_moe_premute,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    use_bf16_gemm_weight_grad=not self.fp8_wgrad,
                    use_auto_subbatch=self.use_auto_subbatch,
                    auto_subbatch_mode=self.auto_subbatch_mode,
                    moe_expert_fusion=self.moe_expert_fusion,
                    moe_subbatch_token_num_after_dispatch=self.moe_subbatch_token_num_after_dispatch,
                    moe_subbatch_diag=self.moe_subbatch_diag,
                    use_ue8m0=self.use_ue8m0,
                    dw_p2p_overlap=self.dw_p2p_overlap,
                    clamp_value=self.config.activation_func_clamp_value,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    use_accuracy_compatible=getattr(
                        self.config, "use_accuracy_compatible", False
                    ),
                    use_w4a8=self.use_w4a8,
                    use_w4a8_fused_quant=self.use_w4a8_fused_quant,
                )

        with profile("combine"):
            hidden_states = self.combine(
                hidden_states,
                combine_overlap_handle=combine_overlap_handle,
                fp8_combine_grad_handle=fp8_combine_grad_handle,
            )

        # Latent MoE: project back from latent space to hidden_size
        if self.use_latent_moe:
            if self.latent_norm is not None:
                hidden_states = self.latent_norm(hidden_states)
            hidden_states = self.fc2_latent_proj(hidden_states)

        return hidden_states

    def compute_gate(
        self, hidden_states, input_ids=None, origin_input_ids=None
    ):
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        return self.gate(
            hidden_states,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )

    def _use_hybrid_ep_fusion(self):
        return self.moe_use_fusion_node and self.use_hybrid_ep_backend

    def _run_hybrid_ep_fusion(
        self,
        dispatched_hidden_states,
        dispatched_probs,
        fp8_dispatched_handle=None,
        is_first_fwd=False,
    ):
        dispatched_hidden_states.stop_gradient = False
        dispatched_probs.stop_gradient = False
        return HybridEPMoePyLayer.apply(
            dispatched_hidden_states,
            dispatched_probs,
            self,
            use_fp8_mlp=self.fp8,
            moe_deep_gemm=self.moe_deep_gemm,
            moe_expert_fusion=self.moe_expert_fusion,
            use_ue8m0=self.use_ue8m0,
            recompute_moe_gate_up=self.recompute_moe_gate_up,
            use_bf16_gemm_weight_grad=not self.fp8_wgrad,
            fp8_dispatched_handle=fp8_dispatched_handle,
            is_first_fwd=is_first_fwd,
            dw_p2p_overlap=self.dw_p2p_overlap,
            clamp_value=self.config.activation_func_clamp_value,
            use_accuracy_compatible=getattr(
                self.config, "use_accuracy_compatible", False
            ),
        )

    def dispatch_preprocess(self, args):
        hidden_states, token_probs, token_indices = args
        if self.use_latent_moe:
            hidden_states = self.fc1_latent_proj(hidden_states)
        assert isinstance(self.token_dispatcher, MoEFlexTokenDispatcher)
        hidden_states = self.token_dispatcher.dispatch_preprocess_overlap(
            hidden_states, token_probs, token_indices
        )
        token_probs = self.token_dispatcher._comm_manager.token_probs
        token_indices = self.token_dispatcher._comm_manager.token_indices
        return hidden_states, token_indices, token_probs

    def compute_dispatch(self, args, async_finish=False):
        hidden_states, token_indices, token_weights = args
        if self.moe_use_fusion_node:
            dispatched_hidden_states, fp8_dispatched_handle = (
                self.token_dispatcher.token_dispatch_overlap(
                    hidden_states,
                    token_indices,
                    token_weights,
                    self.fp8_dispatch,
                    async_finish=async_finish,
                    use_ue8m0=self.use_ue8m0,
                )
            )
            dispatched_probs = (
                self.token_dispatcher._comm_manager.dispatched_probs
            )
            # NOTE: tokens_per_expert_list is stateful and should be saved for recompute.
            tokens_per_expert = (
                self.token_dispatcher._comm_manager.tokens_per_expert
            )
            # dispatched_hidden_states's dtype is fp8, but its gradient's dtype is bf16, so type separation is required; the actual values are passed via a dictionary.
            dispatched_hidden_states, guard_status = GradDtypeGuard.apply(
                dispatched_hidden_states, hidden_states.dtype
            )
            guard_status["x"].stop_gradient = True
            dispatched_indices = None
            if not self._use_hybrid_ep_fusion():
                dispatched_indices = (
                    self.token_dispatcher._comm_manager.dispatched_indices
                )
            return (
                dispatched_hidden_states,
                dispatched_indices,
                dispatched_probs,
                fp8_dispatched_handle,
                tokens_per_expert,
                guard_status,
            )

    def compute_experts(self, args, is_first_fwd=False):
        if self.moe_use_fusion_node:
            (
                dispatched_hidden_states,
                dispatched_indices,
                dispatched_probs,
                fp8_dispatched_handle,
                tokens_per_expert,
                guard_status,
            ) = args
            self.token_dispatcher._comm_manager.tokens_per_expert = (
                tokens_per_expert
            )
            dispatched_hidden_states = GradDtypeUnguard.apply(
                dispatched_hidden_states, guard_status
            )

            if self._use_hybrid_ep_fusion():
                hidden_states = self._run_hybrid_ep_fusion(
                    dispatched_hidden_states,
                    dispatched_probs,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    is_first_fwd=is_first_fwd,
                )
            else:
                hidden_states = FusionMoePyLayer.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices.clone()
                    if is_first_fwd
                    else dispatched_indices,
                    self,
                    self.num_experts_per_tok,
                    use_fp8_mlp=self.fp8,
                    moe_deep_gemm=self.moe_deep_gemm,
                    recompute_moe_gate_up=self.recompute_moe_gate_up,
                    recompute_moe_premute=self.recompute_moe_premute,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    use_bf16_gemm_weight_grad=not self.fp8_wgrad,
                    use_auto_subbatch=self.use_auto_subbatch,
                    auto_subbatch_mode=self.auto_subbatch_mode,
                    moe_expert_fusion=self.moe_expert_fusion,
                    moe_subbatch_token_num_after_dispatch=self.moe_subbatch_token_num_after_dispatch,
                    moe_subbatch_diag=self.moe_subbatch_diag,
                    use_ue8m0=self.use_ue8m0,
                    dw_p2p_overlap=self.dw_p2p_overlap,
                    clamp_value=self.config.activation_func_clamp_value,
                    use_accuracy_compatible=getattr(
                        self.config, "use_accuracy_compatible", False
                    ),
                    use_w4a8=self.use_w4a8,
                    use_w4a8_fused_quant=self.use_w4a8_fused_quant,
                )

            if is_first_fwd:
                hidden_states.stop_gradient = False
        else:
            hidden_states, topk_weights = args
            hidden_states = self.routed_experts_compute(hidden_states)
        return hidden_states

    def compute_combine(self, hidden_states, async_finish=False):
        # Note: RR (use_rr_deepep_combine) is NOT passed here because this method
        # is used by TransformerLayerWithOverlap where shared expert computation is
        # managed by the scheduler separately, not via combine_overlap_handle.
        if self.moe_use_fusion_node:
            hidden_states = self.token_dispatcher._comm_manager.combine(
                hidden_states,
                None,
                async_finish=async_finish,
            )
        else:
            hidden_states = self.combine(hidden_states)
        return hidden_states

    def aux_loss_compute(self, args):
        hidden_states, aux_loss, z_loss, residuals = args
        if self.use_latent_moe:
            if self.latent_norm is not None:
                hidden_states = self.latent_norm(hidden_states)
            hidden_states = self.fc2_latent_proj(hidden_states)
        if self.training and self.router_aux_loss_coef and aux_loss is not None:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(hidden_states, aux_loss)
        else:
            output = hidden_states
        if self.training and z_loss is not None:
            output = AddAuxiliaryLoss.apply(output, z_loss)
        output = output.reshape(residuals.shape)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)[0]
            output = output + shared_output

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        return output

    # ------------------------------------------------------------------
    # Overridable hooks (Template Method pattern)
    # Subclasses (e.g. Gemma4MoELayer) can override these to customize
    # gate/expert input transformation and output post-processing without
    # rewriting the full forward logic.
    # ------------------------------------------------------------------

    def _prepare_gate_input(self, hidden_states, residual):
        """Return the tensor fed into the router. Default: hidden_states."""
        return hidden_states

    def _prepare_expert_input(self, hidden_states, residual):
        """Return the tensor fed into routed experts. Default: hidden_states."""
        return hidden_states

    def _post_routed_output(self, output):
        """Post-process routed expert output before combining with shared. Default: identity."""
        return output

    def _post_shared_output(self, shared_output):
        """Post-process shared expert output before combining. Default: identity."""
        return shared_output

    def _supports_three_path_clone(self) -> bool:
        """Whether the MG-aligned three-path clone applies to this topology.

        The clone assumes router / dispatcher / shared branches all consume the
        same ``hidden_states``. Subclasses overriding the gate/expert input
        hooks (e.g. Gemma4MoELayer routes on ``residual`` and applies
        ``pre_feedforward_layernorm_2``) have a different topology, so the
        clone must not be used there.
        """
        cls = type(self)
        return (
            cls._prepare_gate_input is MoELayer._prepare_gate_input
            and cls._prepare_expert_input is MoELayer._prepare_expert_input
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_ids: paddle.Tensor | None = None,
        residual: paddle.Tensor | None = None,
        origin_input_ids: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """
        Args:
            hidden_states: Shape: [batch_size, seq_len, hidden_size]
            input_ids: Shape: [batch_size, seq_len], optional token ids from embedding input.
            residual: Shape: [batch_size, seq_len, hidden_size], optional separate residual
                      for routing/expert input (used by Gemma4 dual-branch topology).
            origin_input_ids: Shape: [batch_size, seq_len + num_mtp_layers], optional original input_ids.
                Only passed when gpt_model_use_experimental_version is True.

        Returns:
            output: Shape: [batch_size, seq_len, hidden_size]
        """
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            _moe_no_gather = os.environ.get("MODEL_REPRO_MOE_NO_GATHER", "0") == "1"
            self._repro_moe_no_gather = _moe_no_gather
            if not _moe_no_gather:
                hidden_states = GatherOp.apply(hidden_states)
                if residual is not None:
                    residual = GatherOp.apply(residual)
        else:
            self._repro_moe_no_gather = False

        orig_shape = hidden_states.shape
        residuals = hidden_states

        layer_idx = getattr(self, "layer_number", None)

        _three_paths_enabled = (
            getattr(self, "use_accuracy_compatible", False)
            and hidden_states.stop_gradient is False
            and self._supports_three_path_clone()
        )
        if _three_paths_enabled:
            _hs_router_path, _hs_dispatcher_path, _hs_shared_path = (
                ThreePathCloneAlignMG.apply(hidden_states)
            )
            residuals = _hs_shared_path
        else:
            _hs_router_path = _hs_dispatcher_path = hidden_states
        _log_moe_md5(hidden_states, "moe_input", layer_idx)
        if os.environ.get("MODEL_REPRO_MOE_STEP_MD5", "0") == "1":
            import paddle.distributed as _pdmd5

            _mr = _pdmd5.get_rank() if _pdmd5.is_initialized() else 0
            import hashlib as _hb

            _h = hashlib.md5(
                hidden_states.detach().cast("float32").numpy().tobytes()
            ).hexdigest()
            print(f"[MOE-STEP-MD5] r{_mr} layer{layer_idx} hidden {tuple(hidden_states.shape)} md5 {_h}", flush=True)

        self._maybe_pre_allgather_overlap(hidden_states)
        gate_input = self._prepare_gate_input(_hs_router_path, residual)

        (
            capacity,
            topk_weights,
            topk_indices,
            probs,
            mask,
            priorities,
            aux_loss,
            z_loss,
        ) = self.gate(
            gate_input,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )
        from paddlefleet.transformer.multi_latent_attention import _e497_qa_record
        from paddlefleet import utils as _pfutils

        # GatherOp expands [s/tp,b,h] -> [s,b,h]. Hash this TP rank's SP shard
        # so it pairs with torch's ungathered MoE input. Slice is dump-only.
        _tp_size = max(_pfutils.get_pg_size(self.pg_collection.tp), 1)
        _tp_rank = _pfutils.get_pg_rank(self.pg_collection.tp)
        _slen = int(gate_input.shape[0]) // _tp_size
        _lo, _hi = _tp_rank * _slen, (_tp_rank + 1) * _slen
        _e497_qa_record(
            "moeroute",
            gate_input[_lo:_hi],
            topk_weights[_lo:_hi]
            if topk_weights is not None and topk_weights.shape[0] == gate_input.shape[0]
            else topk_weights,
            getattr(self.gate, "e_score_correction_bias", None),
            getattr(self, "layer_number", -1),
            getattr(self, "is_mtp_layer", False),
        )
        # Same-layout vs torch: routing_map/probs are [S, E] after gather; hash
        # the local SP shard so it pairs with torch's ungathered [s/tp, E].
        # Live mask is float32 0/1 (put_along_axis_), not bool — always recast
        # to uint8 so the hash matches torch routing_map.uint8.
        _map = mask[_lo:_hi].cast("uint8")
        _e497_qa_record(
            "moemap",
            gate_input[_lo:_hi],
            _map,
            None,
            getattr(self, "layer_number", -1),
            getattr(self, "is_mtp_layer", False),
        )
        _e497_qa_record(
            "moeprobs",
            gate_input[_lo:_hi],
            probs[_lo:_hi],
            None,
            getattr(self, "layer_number", -1),
            getattr(self, "is_mtp_layer", False),
        )
        if topk_indices is not None and topk_indices.shape[0] == gate_input.shape[0]:
            _e497_qa_record(
                "moetopk",
                gate_input[_lo:_hi],
                topk_indices[_lo:_hi],
                None,
                getattr(self, "layer_number", -1),
                getattr(self, "is_mtp_layer", False),
            )
        # topk_weights, topk_indices: Shape is [seq_len, moe_router_topk]
        # probs: combine weights in [S, E] sparse layout (non-selected positions are 0) [seq_len, num_experts]
        # mask (routing_map): binary selection matrix [seq_len, num_experts]
        # capacity, priorities are used for dropping tokens, currently they are not used

        _log_moe_md5(probs, "probs", layer_idx)
        _log_moe_md5(mask, "routing_mask", layer_idx)
        _router_dump_dir = os.environ.get("MODEL_REPRO_MOE_ROUTER_DUMP_DIR")
        if _router_dump_dir and not getattr(
            MoELayer.forward, "_repro_router_dumped", False
        ):
            MoELayer.forward._repro_router_dumped = True
            import paddle.distributed as _pd

            _rank = _pd.get_rank() if _pd.is_initialized() else 0
            os.makedirs(_router_dump_dir, exist_ok=True)
            try:
                paddle_router_input_r = gate_input if "gate_input" in dir() else hidden_states
                paddle_router_input_r.detach().astype("float32").cpu().numpy().tofile(
                    os.path.join(_router_dump_dir, f"paddle_router_input_r{_rank}.f32.bin")
                )
            except Exception:
                pass
            probs.detach().astype("float32").cpu().numpy().tofile(
                os.path.join(_router_dump_dir, f"paddle_router_probs_r{_rank}.f32.bin")
            )
            mask.detach().astype("uint8").cpu().numpy().tofile(
                os.path.join(_router_dump_dir, f"paddle_routing_map_r{_rank}.u8.bin")
            )
            topk_weights.detach().astype("float32").cpu().numpy().tofile(
                os.path.join(_router_dump_dir, f"paddle_topk_weights_r{_rank}.f32.bin")
            )
            topk_indices.detach().astype("int32").cpu().numpy().tofile(
                os.path.join(_router_dump_dir, f"paddle_topk_indices_r{_rank}.i32.bin")
            )
            with open(
                os.path.join(_router_dump_dir, f"paddle_router_shapes_r{_rank}.txt"), "w"
            ) as _f:
                _f.write(f"probs={tuple(probs.shape)} dtype={probs.dtype}\n")
                _f.write(f"mask={tuple(mask.shape)} dtype={mask.dtype}\n")
                _f.write(
                    f"topk_weights={tuple(topk_weights.shape)} dtype={topk_weights.dtype}\n"
                )
                _f.write(
                    f"topk_indices={tuple(topk_indices.shape)} dtype={topk_indices.dtype}\n"
                )
        if framework._dygraph_tracer()._has_grad:
            log_moe_losses(layer_idx, aux_loss=aux_loss, z_loss=z_loss)

        if (
            self.shared_experts is not None
            and self.moe_shared_expert_overlap
            and self.moe_use_fusion_node
            and self.expert_model_parallel_size > 1
        ):
            combine_overlap_handle = {
                "fn": self.shared_experts,
                "fn_args": (residuals,),
            }
        else:
            combine_overlap_handle = None

        expert_input = self._prepare_expert_input(_hs_dispatcher_path, residual)
        if self.expert_model_parallel_size > 1:
            if self.moe_use_fusion_node:
                output = self.fusion_moe_forward(
                    expert_input,
                    probs,
                    mask,
                    combine_overlap_handle,
                    topk_weights=topk_weights,
                    topk_indices=topk_indices,
                )
            else:
                output = self.custom_forward(
                    expert_input,
                    probs,
                    mask,
                    topk_weights=topk_weights,
                    topk_indices=topk_indices,
                )
        else:
            if len(expert_input.shape) == 3:
                batch_size, seq_len, d_model = expert_input.shape
                reshaped_input = expert_input.reshape([-1, d_model])
            else:
                reshaped_input = expert_input
            # Latent MoE: project to latent space before single-card MoE
            if self.use_latent_moe:
                reshaped_input = self.fc1_latent_proj(reshaped_input)
            if self.moe_expert_fusion:
                if os.environ.get("MODEL_REPRO_MOE_FUSION", "0") == "1":
                    print("[MOE-FUSION-DEBUG] forward takes BRANCH-B (grouped)", flush=True)
                output = self._forward_single_card_grouped_gemm_moe(
                    reshaped_input, mask, probs, topk_indices, topk_weights
                )
            else:
                if os.environ.get("MODEL_REPRO_MOE_FUSION", "0") == "1":
                    print("[MOE-FUSION-DEBUG] forward takes BRANCH-C", flush=True)
                output = self._forward_single_card_moe(
                    reshaped_input, topk_indices, topk_weights
                )
            # Latent MoE: project back from latent space
            if self.use_latent_moe:
                if self.latent_norm is not None:
                    output = self.latent_norm(output)
                output = self.fc2_latent_proj(output)

        _log_moe_md5(output, "moe_routed_output", layer_idx)
        _moe_ds_dump = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR")
        if _moe_ds_dump:
            import paddle.distributed as _pd3

            _dsrank = _pd3.get_rank() if _pd3.is_initialized() else 0
            os.makedirs(_moe_ds_dump, exist_ok=True)
            output.detach().astype("float32").cpu().numpy().tofile(
                os.path.join(
                    _moe_ds_dump,
                    f"paddle_moe_routed_output_l{layer_idx}_r{_dsrank}.f32.bin",
                )
            )

        if self.training and self.router_aux_loss_coef and aux_loss is not None:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        if self.training and z_loss is not None:
            output = AddAuxiliaryLoss.apply(output, z_loss)

        output = output.reshape(orig_shape)
        output = self._post_routed_output(output)
        from paddlefleet.transformer.multi_latent_attention import _e497_qa_record
        from paddlefleet import utils as _pfutils_r

        _tp_size_r = max(_pfutils_r.get_pg_size(self.pg_collection.tp), 1)
        _tp_rank_r = _pfutils_r.get_pg_rank(self.pg_collection.tp)
        _slen_r = int(output.shape[0]) // _tp_size_r
        _lo_r, _hi_r = _tp_rank_r * _slen_r, (_tp_rank_r + 1) * _slen_r
        _e497_qa_record(
            "moerouted",
            residuals[_lo_r:_hi_r],
            output[_lo_r:_hi_r],
            None,
            getattr(self, "layer_number", -1),
            getattr(self, "is_mtp_layer", False),
        )

        if self.shared_experts is not None:
            if combine_overlap_handle is not None:
                shared_output = combine_overlap_handle["fn_out"][0]
            else:
                shared_output = self.shared_experts(residuals)[0]
            from paddlefleet.transformer.multi_latent_attention import _e497_qa_record

            from paddlefleet import utils as _pfutils

            _tp_size = max(_pfutils.get_pg_size(self.pg_collection.tp), 1)
            _tp_rank = _pfutils.get_pg_rank(self.pg_collection.tp)
            _slen = int(residuals.shape[0]) // _tp_size
            _lo, _hi = _tp_rank * _slen, (_tp_rank + 1) * _slen
            _e497_qa_record(
                "moeshared",
                residuals[_lo:_hi],
                shared_output[_lo:_hi],
                None,
                getattr(self, "layer_number", -1),
                getattr(self, "is_mtp_layer", False),
            )
            shared_output = self._post_shared_output(shared_output)
            _moe_ds_dump2 = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR")
            if _moe_ds_dump2:
                import paddle.distributed as _pd4b

                _dsrank2b = _pd4b.get_rank() if _pd4b.is_initialized() else 0
                os.makedirs(_moe_ds_dump2, exist_ok=True)
                residuals.detach().astype("float32").cpu().numpy().tofile(
                    os.path.join(
                        _moe_ds_dump2,
                        f"paddle_shared_input_l{layer_idx}_r{_dsrank2b}.f32.bin",
                    )
                )
                if hasattr(self.shared_experts, "up_gate_proj"):
                    _w1 = self.shared_experts.up_gate_proj.weight
                    _w2 = self.shared_experts.down_proj.weight
                    _w1.detach().astype("float32").cpu().numpy().tofile(
                        os.path.join(_moe_ds_dump2, f"paddle_shared_w1_r{_dsrank2b}.f32.bin")
                    )
                    _w2.detach().astype("float32").cpu().numpy().tofile(
                        os.path.join(_moe_ds_dump2, f"paddle_shared_w2_r{_dsrank2b}.f32.bin")
                    )
                shared_output.detach().astype("float32").cpu().numpy().tofile(
                    os.path.join(
                        _moe_ds_dump2,
                        f"paddle_moe_shared_output_l{layer_idx}_r{_dsrank2b}.f32.bin",
                    )
                )
            output = output + shared_output

        _log_moe_md5(output, "moe_final_output", layer_idx)
        if _moe_ds_dump:
            import paddle.distributed as _pd4

            _dsrank2 = _pd4.get_rank() if _pd4.is_initialized() else 0
            os.makedirs(_moe_ds_dump, exist_ok=True)
            _step_tag = os.environ.get("TRAINER_GLOBAL_STEP", "x")
            output.detach().astype("float32").cpu().numpy().tofile(
                os.path.join(
                    _moe_ds_dump,
                    f"paddle_moe_final_output_l{layer_idx}_s{_step_tag}_r{_dsrank2}.f32.bin",
                )
            )

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            if not getattr(self, "_repro_moe_no_gather", False):
                output = ScatterOp.apply(output)
        return output, None  # None is bias

    def _forward_single_card_moe(
        self,
        hidden_states: paddle.Tensor,
        selected_experts: paddle.Tensor,
        topk_weights: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            selected_experts: TopK experts indices, shape: [seq_len, num_experts_per_tok]
            topk_weights: TopK weights, shape: [seq_len, num_experts_per_tok]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        _, d_model = hidden_states.shape
        _fp32_combine = os.environ.get("MODEL_REPRO_MOE_FP32_COMBINE", "0") == "1"
        _prescale_combine = (
            os.environ.get("MODEL_REPRO_MOE_PRESCALE_COMBINE", "0") == "1"
        )
        final_hidden_states = paddle.zeros_like(
            hidden_states,
            dtype="float32" if _fp32_combine else hidden_states.dtype,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).transpose([2, 1, 0])
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(
            axis=-1
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, d_model])
            current_weight = topk_weights[idx, top_x].unsqueeze(-1)
            if _prescale_combine:
                # E-112 probe: mcore folds the routing probability into the
                # ACTIVATION before fc2 (Megatron experts.py:783-788), whereas the
                # default paddle path multiplies expert_out AFTER fc2. The two are
                # algebraically identical but round differently in bf16. Pass the
                # weight down as per_token_scale so MLP.forward applies it in the
                # mcore position, and skip the post-fc2 multiply here.
                expert_out = expert_layer(current_state, per_token_scale=current_weight)[0]
                if _fp32_combine:
                    # E-113 probe: prescale (mcore multiply position) PLUS fp32
                    # accumulation of the already-weighted expert contributions.
                    final_hidden_states_tmp = paddle.zeros_like(final_hidden_states)
                    final_hidden_states_tmp = paddle.scatter(
                        final_hidden_states_tmp,
                        idx.reshape([-1]),
                        expert_out.cast("float32"),
                        overwrite=False,
                    )
                    final_hidden_states = final_hidden_states + final_hidden_states_tmp
                    continue
                current_hidden_states = expert_out
            else:
                expert_out = expert_layer(current_state)[0]
                if _fp32_combine:
                    # E-099 probe: accumulate the weighted expert contributions in fp32
                    # (mirrors mcore's fp32 unpermute/combine) instead of bf16.
                    current_hidden_states = expert_out.cast("float32") * current_weight.cast(
                        "float32"
                    )
                    final_hidden_states_tmp = paddle.zeros_like(final_hidden_states)
                    final_hidden_states_tmp = paddle.scatter(
                        final_hidden_states_tmp,
                        idx.reshape([-1]),
                        current_hidden_states,
                        overwrite=False,
                    )
                    final_hidden_states = final_hidden_states + final_hidden_states_tmp
                    continue
                current_hidden_states = expert_out * current_weight

            # use scatter to replace index_add
            final_hidden_states_tmp = paddle.zeros_like(final_hidden_states)
            final_hidden_states_tmp = paddle.scatter(
                final_hidden_states_tmp,
                idx.reshape([-1]),
                current_hidden_states.to(hidden_states.dtype),
                overwrite=False,
            )
            final_hidden_states = final_hidden_states + final_hidden_states_tmp
        return final_hidden_states.cast(hidden_states.dtype)

    def _forward_single_card_grouped_gemm_moe(
        self,
        hidden_states: paddle.Tensor,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_indices: paddle.Tensor | None = None,
        topk_weights: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            routing_map: Routing map, shape: [seq_len, num_experts]
            probs: Probabilities of selecting each expert, shape: [seq_len, num_experts]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        def _convert_routing_map_and_probs(
            routing_map: paddle.Tensor, probs: paddle.Tensor, topk: int
        ):
            routing_map = routing_map.astype("bool")
            masked_probs = probs * routing_map.astype("float32")
            weights, indices = paddle.topk(masked_probs, k=topk, axis=-1)
            return indices, weights

        if self.using_sonic_moe:
            use_fp8 = self.fp8 is not None
            final_hidden_states = self.grouped_gemm_experts(
                hidden_states,
                topk_indices,
                topk_weights,
                use_fp8,
                recompute_moe_gate_up=self.recompute_moe_gate_up,
            )
            return final_hidden_states.cast(hidden_states.dtype)
        else:
            # E-118: LOCAL-SHARD mode. mcore runs the routed experts on the
            # rank-local SP shard (token slice of this TP rank) and never gathers
            # for the experts; paddle's wrapper gathers to 60 and would run the
            # union, which changes every per-expert GEMM M and (for bf16) the row
            # results. Here we run permute/GEMM/unpermute on the LOCAL slice and
            # place the result back at this rank's slot; router/shared still use
            # the gathered 60 (matching mcore's shared-expert allgather semantics).
            # Accuracy-compatible expert path (E-163 / E-256). These behaviours
            # are what make the routed-expert block bit-identical to Megatron-LM;
            # they are enabled together by ``config.use_accuracy_compatible`` and
            # are not independent tuning knobs.
            #
            #   _use_ac            permute/unpermute use the MG-aligned
            #                      gather/sum PyLayers (fp32 topk reduction).
            #                      Previously this was gated only on
            #                      MODEL_REPRO_MOE_AC_ALIGNED, so the formal
            #                      accuracy-compatible path still ran default
            #                      index_select / scatter unpermute.
            #   _prescale_combine  fold the routing probs into the post-GLU
            #                      activation BEFORE fc2 (Megatron
            #                      moe/experts.py:786-788) and unpermute without
            #                      probs, instead of scaling after fc2.
            #   _shard_gemm        batch each expert's GEMM per sequence-parallel
            #                      shard, matching the M that mcore's per-rank
            #                      expert calls see.
            #
            # ``MODEL_REPRO_MOE_AC_ALIGNED`` / ``MODEL_REPRO_MOE_PRESCALE_COMBINE``
            # / ``MODEL_REPRO_MOE_SHARD_GEMM`` remain honoured so the behaviours
            # can still be exercised in isolation while the flag is off.
            _ac_expert_path = getattr(self.config, "use_accuracy_compatible", False)
            _use_ac = (
                _ac_expert_path
                or os.environ.get("MODEL_REPRO_MOE_AC_ALIGNED", "0") == "1"
            )
            _prescale_combine = (
                _ac_expert_path
                or os.environ.get("MODEL_REPRO_MOE_PRESCALE_COMBINE", "0") == "1"
            )
            _shard_gemm = (
                _ac_expert_path
                or os.environ.get("MODEL_REPRO_MOE_SHARD_GEMM", "0") == "1"
            )
            # LOCAL_SHARD remains an explicit env gate only. Enabling it from
            # use_accuracy_compatible (tried in E-256) sliced expert wgrad to
            # the local 30-token SP shard and broke rank2==rank3 equality that
            # mcore keeps (expert weights are not sequence-parallel). Permute
            # on the gathered 60 plus per-shard GEMM (_shard_gemm) is the
            # accuracy-compatible path.
            if (
                os.environ.get("MODEL_REPRO_MOE_LOCAL_SHARD", "0") == "1"
                and self.sequence_parallel
                and self.pg_collection is not None
            ):
                from paddlefleet import utils as _pfutils

                _tp_size = max(_pfutils.get_pg_size(self.pg_collection.tp), 1)
                _tp_rank = _pfutils.get_pg_rank(self.pg_collection.tp)
                _n_tok = hidden_states.shape[0]
                _slen = _n_tok // _tp_size
                _lo, _hi = _tp_rank * _slen, (_tp_rank + 1) * _slen
                _hs_loc = hidden_states[_lo:_hi]
                _rm_loc = routing_map[_lo:_hi]
                _pr_loc = probs[_lo:_hi] if probs is not None else None
                _tpe_loc = _rm_loc.sum(axis=0)
                if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                    import paddle.distributed as _hld
                    _hlk = _hld.get_rank() if _hld.is_initialized() else 0
                    _hdir = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or ""
                    if _hdir:
                        os.makedirs(_hdir, exist_ok=True)
                        _hs_loc.detach().astype("float32").cpu().numpy().tofile(
                            os.path.join(_hdir, f"paddle_hs_loc_r{_hlk}.f32.bin")
                        )
                _perm_loc, _sorted_loc = permute(
                    _hs_loc, _rm_loc, _tpe_loc, use_accuracy_compatible=_use_ac
                )
                if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                    import paddle.distributed as _ppd
                    _ppk = _ppd.get_rank() if _ppd.is_initialized() else 0
                    _ppdir = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or ""
                    _pplay = getattr(self, "layer_number", None)
                    if _ppdir:
                        os.makedirs(_ppdir, exist_ok=True)
                        _perm_loc.detach().astype("float32").cpu().numpy().tofile(
                            os.path.join(_ppdir, f"paddle_perm_loc_l{_pplay}_r{_ppk}.f32.bin")
                        )
                        _sorted_loc.detach().cpu().numpy().tofile(
                            os.path.join(_ppdir, f"paddle_sorted_loc_l{_pplay}_r{_ppk}.i64.bin")
                        )
                _pp_loc = None
                if _pr_loc is not None:
                    _pp_loc = _pr_loc.T.contiguous().masked_select(
                        _rm_loc.T.contiguous().cast("bool")
                    )
                if os.environ.get("MODEL_REPRO_MOE_EXPERT_DUMP_DIR"):
                    import paddle.distributed as _pdl2

                    _dc = _pdl2.get_rank() if _pdl2.is_initialized() else 0
                    os.makedirs(
                        os.environ["MODEL_REPRO_MOE_EXPERT_DUMP_DIR"], exist_ok=True
                    )
                    _tpe_loc.detach().cpu().numpy().tofile(
                        os.path.join(
                            os.environ["MODEL_REPRO_MOE_EXPERT_DUMP_DIR"],
                            f"local_tpe_r{_dc}.i64.bin",
                        )
                    )
                    _pr_loc.detach().astype("float32").cpu().numpy().tofile(
                        os.path.join(
                            os.environ["MODEL_REPRO_MOE_EXPERT_DUMP_DIR"],
                            f"local_probs_r{_dc}.f32.bin",
                        )
                    )
                    _rm_np = _rm_loc.detach().cpu().numpy()
                    print(
                        f"[LOCAL-SHARD-DEBUG] r{_dc} tpe={_tpe_loc.detach().cpu().tolist()} "
                        f"rm_np_dtype={_rm_np.dtype} shape={_rm_np.shape} "
                        f"nonzero={int((_rm_np != 0).sum())} "
                        f"probs_dtype={_pr_loc.dtype} pp_dtype={None if _pp_loc is None else _pp_loc.dtype} "
                        f"pp_first={None if _pp_loc is None else _pp_loc.detach().cast('float32').cpu().numpy()[:4].tolist()}",
                        flush=True,
                    )
                if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_FORCE", "0") == "1":
                    # E-171: bypass the fused/grouped GEMM and run a plain
                    # per-expert matmul pipeline (fc1 bf16 -> silu -> *probs ->
                    # fc2 bf16) so the kernel differences of the fused group
                    # GEMM are excluded from the cross-frame diff compare.
                    import paddle.nn.functional as _pf

                    _exps = self.grouped_gemm_experts
                    _w1 = _exps.weight1  # [E, H, 2*inter]
                    _w2 = _exps.weight2  # [E, 2*inter? , H]
                    _rows = []
                    _tpe_i32 = _tpe_loc.astype("int32")
                    _starts = paddle.cumsum(paddle.concat([paddle.zeros([1], dtype="int32"), _tpe_i32], 0), 0)
                    for _e in range(int(_tpe_loc.shape[0])):
                        _m = int(_tpe_loc[_e])
                        if _m == 0:
                            continue
                        _seg = _perm_loc[_starts[_e] : _starts[_e + 1]]
                        if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                            import paddle.distributed as _ped
                            _perk = _ped.get_rank() if _ped.is_initialized() else 0
                            _pdir = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or os.environ.get("MODEL_REPRO_MOE_EXPERT_DUMP_DIR")
                            if _pdir:
                                os.makedirs(_pdir, exist_ok=True)
                                _seg.detach().astype("float32").cpu().numpy().tofile(
                                    os.path.join(_pdir, f"paddle_fc1_in_e{_e}_r{_perk}.f32.bin")
                                )
                        _h1 = _pf.linear(_seg, _w1[_e])
                        if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                            import paddle.distributed as _pe
                            _per = _pe.get_rank() if _pe.is_initialized() else 0
                            _dir = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or os.environ.get("MODEL_REPRO_MOE_EXPERT_DUMP_DIR")
                            _play = getattr(self, "layer_number", None)
                            if _dir:
                                os.makedirs(_dir, exist_ok=True)
                                _h1.detach().astype("float32").cpu().numpy().tofile(
                                    os.path.join(_dir, f"paddle_fc1_raw_l{_play}_e{_e}_r{_per}.f32.bin")
                                )
                        _g, _l = paddle.chunk(_h1, 2, axis=-1)
                        if os.environ.get("MODEL_REPRO_MOE_GLU_EXPLICIT", "0") == "1":
                            # E-184: torch F.silu == fp32 silu then cast bf16
                            # (verified 0 diff). Compute silu in fp32, cast bf16,
                            # then bf16 multiply with _l (matches torch order).
                            _act = (
                                _g.cast("float32")
                                * paddle.nn.functional.sigmoid(_g.cast("float32"))
                            ).cast("bfloat16") * _l
                        else:
                            _act = paddle.nn.functional.silu(_g) * _l
                        if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                            import paddle.distributed as _gld2
                            _glr2 = _gld2.get_rank() if _gld2.is_initialized() else 0
                            _gldir2 = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or ""
                            _gll2 = getattr(self, "layer_number", None)
                            if _gldir2:
                                os.makedirs(_gldir2, exist_ok=True)
                                _act.detach().astype("float32").cpu().numpy().tofile(
                                    os.path.join(_gldir2, f"paddle_gluout_l{_gll2}_e{_e}_r{_glr2}.f32.bin")
                                )
                        if _pp_loc is not None:
                            if os.environ.get("MODEL_REPRO_MOE_PP_MUL_TORCH", "0") == "1":
                                # torch: act(bf16)*probs(f32) -> f32 -> cast bf16
                                _act = (
                                    _act.cast("float32")
                                    * _pp_loc[_starts[_e] : _starts[_e + 1]].unsqueeze(-1)
                                ).cast("bfloat16")
                            else:
                                if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1" and not getattr(
                                    MoELayer, "_ppdtype_dbg", False
                                ):
                                    MoELayer._ppdtype_dbg = True
                                    _ppseg = _pp_loc[_starts[_e] : _starts[_e + 1]].unsqueeze(-1)
                                    print(
                                        f"[PPDBG] _act.dtype={_act.dtype} _pp.dtype={_ppseg.dtype} "
                                        f"_pr.dtype={_pr_loc.dtype if _pr_loc is not None else None}",
                                        flush=True,
                                    )
                                _act = _act * _pp_loc[_starts[_e] : _starts[_e + 1]].unsqueeze(-1)
                                if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1" and not getattr(
                                    MoELayer, "_ppres_dbg", False
                                ):
                                    MoELayer._ppres_dbg = True
                                    print(f"[PPRESDBG] after-mul _act.dtype={_act.dtype} shape={tuple(_act.shape)}", flush=True)
                        if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                            import paddle.distributed as _acd
                            _acr = _acd.get_rank() if _acd.is_initialized() else 0
                            _acdir = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or ""
                            _aclay = getattr(self, "layer_number", None)
                            if _acdir:
                                os.makedirs(_acdir, exist_ok=True)
                                _act.detach().astype("float32").cpu().numpy().tofile(
                                    os.path.join(_acdir, f"paddle_act_l{_aclay}_e{_e}_r{_acr}.f32.bin")
                                )
                        _h1_act = _act
                        _h2 = _pf.linear(_act, _w2[_e])
                        _h2 = paddle.nn.functional.linear(_act, _w2[_e])
                        if os.environ.get("MODEL_REPRO_MOE_PER_EXPERT_DUMP", "0") == "1":
                            import paddle.distributed as _f2d
                            _f2r = _f2d.get_rank() if _f2d.is_initialized() else 0
                            _f2dir = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR") or ""
                            _f2l = getattr(self, "layer_number", None)
                            if _f2dir:
                                os.makedirs(_f2dir, exist_ok=True)
                                _h2.detach().astype("float32").cpu().numpy().tofile(
                                    os.path.join(_f2dir, f"paddle_fc2out_l{_f2l}_e{_e}_r{_f2r}.f32.bin")
                                )
                        _rows.append(_h2)
                    _out_expert = paddle.concat(_rows, axis=0) if _rows else paddle.zeros(
                        [0, self.hidden_size], dtype=_perm_loc.dtype
                    )
                else:
                    _out_expert = self.grouped_gemm_experts(
                        _perm_loc, _tpe_loc, permuted_probs=_pp_loc
                    )[0]
                if os.environ.get("MODEL_REPRO_MOE_EXPERT_DUMP_DIR") and os.environ.get("MODEL_REPRO_MOE_UNPERM_EXPERT_ORDER", "0") == "1":
                    import paddle.distributed as _pdl3

                    _dc3 = _pdl3.get_rank() if _pdl3.is_initialized() else 0
                    _out_expert.detach().astype("float32").cpu().numpy().tofile(
                        os.path.join(
                            os.environ["MODEL_REPRO_MOE_EXPERT_DUMP_DIR"],
                            f"weighted_expert_rows_r{_dc3}.f32.bin",
                        )
                    )
                _out_final = unpermute(
                    _out_expert,
                    _sorted_loc,
                    restore_shape=[_slen, self.hidden_size],
                    probs=None,
                    routing_map=_rm_loc if _use_ac else None,
                    use_accuracy_compatible=_use_ac,
                )
                if os.environ.get("MODEL_REPRO_MOE_UNPERM_EXPERT_ORDER", "0") == "1" and os.environ.get("MODEL_REPRO_MOE_UNPERM_VERBOSE", "0") == "1":
                    import paddle.distributed as _pv
                    _vr = _pv.get_rank() if _pv.is_initialized() else 0
                    print(f"[UNPERM-VERBOSE] r{_vr} gathersum rm={_out_final.shape} tpe={_tpe_loc.cast('int32').numpy().tolist()[:4]}...", flush=True)
                _out_loc = _out_final.cast(hidden_states.dtype)
                _out_full = paddle.zeros_like(hidden_states)
                _out_full[_lo:_hi] = _out_loc
                return _out_full

            tokens_per_expert = routing_map.sum(axis=0)
            permuted_local_hidden_states, sorted_indices = permute(
                hidden_states,
                routing_map,
                tokens_per_expert,
                use_accuracy_compatible=_use_ac,
            )
            _idx_dump = os.environ.get("MODEL_REPRO_UNPERM_INDEX_DUMP")
            if _idx_dump:
                import paddle.distributed as _pdix

                _ir = _pdix.get_rank() if _pdix.is_initialized() else 0
                _lidx = getattr(self, "layer_number", "x")
                os.makedirs(_idx_dump, exist_ok=True)
                sorted_indices.detach().cpu().numpy().astype("int64").tofile(
                    os.path.join(_idx_dump, f"paddle_sorted_indices_l{_lidx}_r{_ir}.i64.bin")
                )
                routing_map.cast("uint8").detach().cpu().numpy().tofile(
                    os.path.join(_idx_dump, f"paddle_routing_map_l{_lidx}_r{_ir}.u8.bin")
                )
                tokens_per_expert.cast("int64").detach().cpu().numpy().tofile(
                    os.path.join(_idx_dump, f"paddle_tpe_l{_lidx}_r{_ir}.i64.bin")
                )
                with open(
                    os.path.join(_idx_dump, f"paddle_unperm_meta_l{_lidx}_r{_ir}.txt"),
                    "w",
                ) as _mf:
                    _mf.write(
                        f"hidden={tuple(hidden_states.shape)} perm={tuple(permuted_local_hidden_states.shape)} "
                        f"si={tuple(sorted_indices.shape)} rm={tuple(routing_map.shape)} "
                        f"use_ac={_use_ac} tp={self.tensor_model_parallel_size}\n"
                    )
            # E-107/E-115: mcore batches each expert GEMM on the rank-local SP
            # shard's tokens; paddle's permute output holds the union of all TP
            # shards. row_owner maps each permuted row to its shard id so the
            # expert layer can re-group GEMMs per shard (M alignment).
            #
            # Under ``use_accuracy_compatible`` this is required for numerical
            # equivalence, not merely an optimization: bf16 GEMM rows are not
            # M-invariant, so running one GEMM over the 60-token union produces
            # different rows than mcore's two 30-token per-shard GEMMs even
            # though the inputs are bit-identical.
            _row_owner = None
            if _shard_gemm:
                _shard_len = max(
                    hidden_states.shape[0] // max(self.tensor_model_parallel_size, 1),
                    1,
                )
                _row_owner = sorted_indices // _shard_len
            if _prescale_combine:
                # E-114 probe: mcore folds the routing probs into the post-GLU
                # activation BEFORE fc2 (Megatron experts.py:787-788), and then
                # unpermutes WITHOUT probs so the fp32 accumulate path runs
                # (moe_utils._unpermute_fp32_accum). PaddleFleet's default
                # branch (b) multiplies probs AFTER fc2 inside unpermute
                # (ApplyPermutedProbs) and then scatters; the two are algebraically
                # identical but round differently in bf16.
                _permuted_probs = None
                if probs is not None:
                    _permuted_probs = probs.T.contiguous().masked_select(
                        routing_map.T.contiguous().cast("bool")
                    )
                grouped_expert_out = self.grouped_gemm_experts(
                    permuted_local_hidden_states,
                    tokens_per_expert,
                    permuted_probs=_permuted_probs,
                    row_owner=_row_owner,
                )[0]
                # Keep routing_map even when probs are already folded: the
                # accuracy-compatible unpermute uses it to rebuild the
                # token-major gather index (Megatron token_dispatcher.py
                # passes routing_map into unpermute with merging_probs=None).
                # Passing None dropped through to scatter_add and skipped the
                # aligned gather-sum backward (E-256).
                final_hidden_states = unpermute(
                    grouped_expert_out,
                    sorted_indices,
                    restore_shape=hidden_states.shape,
                    probs=None,
                    routing_map=routing_map if _use_ac else None,
                    use_accuracy_compatible=_use_ac,
                )
            else:
                grouped_expert_out = self.grouped_gemm_experts(
                    permuted_local_hidden_states,
                    tokens_per_expert,
                    row_owner=_row_owner,
                )[0]
                final_hidden_states = unpermute(
                    grouped_expert_out,
                    sorted_indices,
                    restore_shape=hidden_states.shape,
                    probs=probs,
                    routing_map=routing_map,
                    use_accuracy_compatible=_use_ac,
                )
            return final_hidden_states.cast(hidden_states.dtype)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if not (self.moe_use_fusion_node and self.fp8):
            return
        if hasattr(self, "grouped_gemm_experts") and isinstance(
            self.grouped_gemm_experts, SonicMoEExpert
        ):
            self.grouped_gemm_experts.quant_weight()
            return

        def quantize_weights(
            weight_list, weight_obj=None, quant_transpose=None
        ):
            """Helper function to quantize a list of weights."""
            if weight_obj is None:
                weight_obj = weight_list[0]

            # 始终量化非转置版（行为对齐，fp8_weight_stacked 始终存在）
            fp8_weight, fp8_scale = fused_stack_quant_without_cache(
                weight_list, transpose=False, use_ue8m0=self.use_ue8m0
            )
            weight_obj.fp8_weight_stacked = fp8_weight
            weight_obj.fp8_scale_stacked = fp8_scale

            if quant_transpose is None or quant_transpose is True:
                fp8_weight_t, fp8_scale_t = fused_stack_quant_without_cache(
                    weight_list, transpose=True, use_ue8m0=self.use_ue8m0
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            else:
                weight_obj.fp8_weight_stacked_transpose = None
                weight_obj.fp8_scale_stacked_transpose = None
                if self.use_ue8m0:
                    from paddlefleet.triton_ops import (
                        fuse_stack_ue8m0_scale_transpose,
                    )

                    converted_scale = fuse_stack_ue8m0_scale_transpose(
                        fp8_scale,
                        len(weight_list),
                        weight_list[0].shape[0],
                        weight_list[0].shape[1],
                    )
                    weight_obj.fp8_scale_stacked_transpose = converted_scale

        if hasattr(self, "grouped_gemm_experts"):
            if batch_mode:
                expert_w1 = self.grouped_gemm_experts.weight1
                expert_w2 = self.grouped_gemm_experts.weight2
                local_expert_num = expert_w1.shape[0]
                expert_w1_list = [
                    expert_w1[i, :, :] for i in range(local_expert_num)
                ]
                expert_w2_list = [
                    expert_w2[i, :, :] for i in range(local_expert_num)
                ]

                # Batch mode: process all experts' weights together
                if expert_w1_list:
                    quantize_weights(
                        expert_w1_list,
                        self.grouped_gemm_experts.weight1,
                        quant_transpose,
                    )
                if expert_w2_list:
                    quantize_weights(
                        expert_w2_list,
                        self.grouped_gemm_experts.weight2,
                        quant_transpose,
                    )

            else:
                raise NotImplementedError(
                    "Not support individual mode for fuse_expert_fp8_weight_quant yet."
                )

            return

        if batch_mode:
            # Batch mode: process all experts' weights together
            expert_w1_list = [
                expert.up_gate_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            expert_w2_list = [
                expert.down_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            if expert_w1_list:
                quantize_weights(
                    expert_w1_list, expert_w1_list[0], quant_transpose
                )
            if expert_w2_list:
                quantize_weights(
                    expert_w2_list, expert_w2_list[0], quant_transpose
                )

        else:
            # Individual mode: process each expert's weights separately
            for expert in self.experts:
                if expert is not None:
                    quantize_weights(
                        [expert.up_gate_proj.weight],
                        quant_transpose=quant_transpose,
                    )
                    quantize_weights(
                        [expert.down_proj.weight],
                        quant_transpose=quant_transpose,
                    )

    def clear_fp8_quant_weight(self):
        """Clear cached FP8 quantized weights to release memory."""

        logger.info(
            "Clearing FP8 quantized weights in MoE layer: "
            "[fp8_weight_stacked, fp8_scale_stacked, "
            "fp8_weight_stacked_transpose, fp8_scale_stacked_transpose]"
        )

        if not (self.moe_use_fusion_node and self.fp8):
            return

        fp8_attrs = (
            "fp8_weight_stacked",
            "fp8_scale_stacked",
            "fp8_weight_stacked_transpose",
            "fp8_scale_stacked_transpose",
        )

        def _clear_attrs(weight_obj):
            for attr in fp8_attrs:
                if hasattr(weight_obj, attr):
                    delattr(weight_obj, attr)

        if hasattr(self, "grouped_gemm_experts"):
            if isinstance(self.grouped_gemm_experts, SonicMoEExpert):
                self.grouped_gemm_experts.clear_fp8_weights()
            else:
                _clear_attrs(self.grouped_gemm_experts.weight1)
                _clear_attrs(self.grouped_gemm_experts.weight2)
        else:
            for expert in self.experts:
                if expert is not None:
                    _clear_attrs(expert.up_gate_proj.weight)
                    _clear_attrs(expert.down_proj.weight)

    def use_fp8(self):
        if self.moe_use_fusion_node and self.fp8:
            return True
        return False

    def set_layer_number(self, layer_number, is_mtp_layer: bool = False):
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        experts = getattr(self, "grouped_gemm_experts", None)
        if experts is not None:
            experts.layer_number = layer_number
            experts.is_mtp_layer = is_mtp_layer
        # Assign routed-expert 'color' now that the layer number is known. This
        # is the single place color is set for experts (Paddle forbids
        # reassigning it): the MTP-shared last layer uses the no-hook color.
        self._color_expert_params()
        assert hasattr(self.gate, "set_layer_number"), (
            "expect gate has method 'set_layer_number'"
        )
        # Hash routing activation (moe_n_hash_layers) is decided by the router
        # itself based on layer_number. See TopKRouter._setup_hash_layer.
        self.gate.set_layer_number(layer_number, is_mtp_layer=is_mtp_layer)

    def _color_expert_params(self):
        """Set the sharding 'color' on routed-expert params (called once).

        Only needed when ``mtp_shared_last_layer`` is enabled: in that case the
        expert params were intentionally left uncolored at construction (the
        moe_expert vs no-hook choice depends on the layer number). Picks
        ``moe_weight_no_hook`` for the MTP-shared backbone last layer so the
        sharding-stage1 optimizer reduces those shared params synchronously (no
        overlap hook); otherwise the normal ``moe_expert`` color is used.

        Params already colored at construction (the common, non-shared-MTP case)
        are skipped: Paddle forbids reassigning a non-None color, and their
        color would be ``moe_expert`` either way.
        """
        if self.expert_model_parallel_size <= 1:
            return
        # Lazy import to avoid a circular import: transformer_layer imports
        # MoELayer from this module.
        from paddlefleet.transformer.transformer_layer import (
            is_mtp_shared_last_layer,
        )

        fusion_experts = getattr(self, "grouped_gemm_experts", None)
        if fusion_experts is not None:
            expert_params = fusion_experts.parameters()
        else:
            assert self.experts is not None, "experts should be initialized."
            expert_params = self.experts.parameters()
        color_key = (
            "moe_weight_no_hook"
            if is_mtp_shared_last_layer(
                self.config, self.layer_number, self.is_mtp_layer
            )
            else "moe_expert"
        )
        for p in expert_params:
            # Skip params already colored at construction (the non-shared-MTP
            # case); Paddle forbids reassigning a non-None color. Uncolored
            # params carry no color attribute (None) or the -1 sentinel.
            color = getattr(p, "color", None)
            if color not in (None, -1):
                continue
            p.color = {"color": color_key, "group": self.moe_grad_group}


class Gemma4TopKRouter(TopKRouter):
    """Gemma4 MoE router aligned with ms-swift/HF Gemma4TextRouter.

    Reuses TopKRouter for padding mask, SP/CP, aux/z loss handling.
    Only adds:
    1. Input normalization: scaleless RMSNorm + learned scale * (1/sqrt(d))
    2. Config overrides: softmax scoring, norm_topk_prob, learnable per-expert scale
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        # Use a shallow copy to avoid polluting the shared config object.
        import copy

        config = copy.copy(config)
        # Configure TopKRouter to match Gemma4 behavior
        config.scoring_func = "softmax"
        config.norm_topk_prob = True
        config.topk_method = "greedy"
        config.routed_scaling_factor_learnable = True
        config.routed_scaling_factor = 1.0
        config.router_aux_loss_coef = 0.0
        config.router_z_loss_coef = 0.0
        # Greedy topk is incompatible with moe_topk_fusion (requires e_score_correction_bias
        # which is only created for topk_method == "noaux_tc").
        config.moe_topk_fusion = False
        super().__init__(config, pg_collection)

        # Gemma4-specific: input normalization scale (learnable, aligned with HF nn.Parameter)
        hidden_size = config.hidden_size
        self.router_input_scale = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self._inv_sqrt_d = hidden_size**-0.5

    def _normalize_input(self, hidden_states):
        """Scaleless RMSNorm + learned scale."""
        h = hidden_states.cast("float32")
        rms = (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        h = (h / rms).cast(hidden_states.dtype)
        return h * self.router_input_scale * self._inv_sqrt_d

    def forward(self, input, input_ids=None, origin_input_ids=None):
        """Normalize input, then delegate to TopKRouter for full routing logic."""
        normalized_input = self._normalize_input(input)
        return super().forward(
            normalized_input,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )


class Gemma4MoELayer(MoELayer):
    """Gemma4 MoE via base-class hooks (no forward override).

    Customizations over base MoELayer:
      - Gate: Gemma4TopKRouter (internal RMS norm + per_expert_scale)
      - Activation: GeGLU (gelu_tanh(gate) * up)
      - Dual-branch topology via hooks:
        * _prepare_gate_input  → route on residual
        * _prepare_expert_input → pre_feedforward_layernorm_2(residual)
        * _post_routed_output  → post_moe_layernorm
        * _post_shared_output  → post_shared_expert_layernorm

    Norms (aligned with HF naming):
      - post_shared_expert_layernorm (= HF post_feedforward_layernorm_1)
      - pre_feedforward_layernorm_2 (= HF pre_feedforward_layernorm_2)
      - post_moe_layernorm (= HF post_feedforward_layernorm_2)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MoESublayers | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        if (
            not hasattr(config, "n_shared_experts")
            or config.n_shared_experts is None
        ):
            config.n_shared_experts = 1
        super().__init__(config, sublayers, pg_collection)

        self.gate = Gemma4TopKRouter(config=config, pg_collection=pg_collection)

        shared_size = getattr(
            config, "moe_shared_expert_intermediate_size", None
        )
        if shared_size and self.shared_experts is not None:
            shared_expert_config = deepcopy(config)
            shared_expert_config.intermediate_size = shared_size
            self.shared_experts = StandardMLPSharedExpert(
                config=shared_expert_config,
                moe_intermediate_size=shared_size,
                is_expert=False,
                mlp_spec=self.moe_sublayers.mlp_spec,
            )

        self._activation_type = "geglu"

        if (
            hasattr(self, "grouped_gemm_experts")
            and self.grouped_gemm_experts is not None
        ):
            gelu_tanh = functools.partial(F.gelu, approximate=True)

            def _gemma4_glu(x):
                x = paddle.chunk(x, 2, dim=-1)
                return gelu_tanh(x[0]) * x[1]

            self.grouped_gemm_experts.activation_func = _gemma4_glu
            self.grouped_gemm_experts.config.hidden_act = gelu_tanh

        from paddlefleet.transformer.paddle_norm import RMSNorm

        self.post_shared_expert_layernorm = RMSNorm(config)
        self.pre_feedforward_layernorm_2 = RMSNorm(config)
        self.post_moe_layernorm = RMSNorm(config)

        if (
            hasattr(self, "grouped_gemm_experts")
            and self.grouped_gemm_experts is not None
        ):
            import types

            from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
                build_sharded_state_dict,
                shard_weight,
            )

            grouped = self.grouped_gemm_experts

            def _gemma4_grouped_sharded_state_dict(
                self_inner, structured_name_prefix=""
            ):
                state_dict = self_inner.state_dict(structured_name_prefix="")
                sharded_dict = {}
                full_key1 = f"{structured_name_prefix}weight1"
                full_key2 = f"{structured_name_prefix}weight2"
                if self_inner.ep_group is None:
                    sharded_dict = build_sharded_state_dict(
                        state_dict, None, structured_name_prefix
                    )
                else:
                    sharded_dict[full_key1] = shard_weight(
                        key=full_key1,
                        weight=state_dict["weight1"],
                        axis=0,
                        group=self_inner.ep_group,
                    )
                    sharded_dict[full_key1].grouped_gemm_param = True
                    sharded_dict[full_key2] = shard_weight(
                        key=full_key2,
                        weight=state_dict["weight2"],
                        axis=0,
                        group=self_inner.ep_group,
                    )
                    sharded_dict[full_key2].grouped_gemm_param = True
                return sharded_dict

            grouped.sharded_state_dict = types.MethodType(
                _gemma4_grouped_sharded_state_dict, grouped
            )

    # ------------------------------------------------------------------
    # Hook overrides: dual-branch topology (shared from hidden_states,
    # routed from residual with extra norms)
    # ------------------------------------------------------------------

    def _prepare_gate_input(self, hidden_states, residual):
        """Route on residual (Gemma4TopKRouter applies internal normalization)."""
        return residual if residual is not None else hidden_states

    def _prepare_expert_input(self, hidden_states, residual):
        """Apply pre_feedforward_layernorm_2 to residual before expert compute."""
        src = residual if residual is not None else hidden_states
        return self.pre_feedforward_layernorm_2(src)

    def _post_routed_output(self, output):
        """Apply post_moe_layernorm after routed expert combine."""
        return self.post_moe_layernorm(output)

    def _post_shared_output(self, shared_output):
        """Apply post_shared_expert_layernorm to shared expert output."""
        return self.post_shared_expert_layernorm(shared_output)
