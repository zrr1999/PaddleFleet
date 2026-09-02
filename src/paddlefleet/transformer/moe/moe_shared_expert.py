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


from copy import deepcopy

import os

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig


class StandardMLPSharedExpert(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        moe_intermediate_size: int,
        is_expert: bool,
        mlp_spec: MLPSublayersSpec,
        tp_group=None,
    ):
        if moe_intermediate_size == config.intermediate_size:
            super().__init__(
                config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                tp_group=tp_group,
            )
        else:
            # Local SequentialMLP can still be used here by overriding the intermediate_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                tp_group=tp_group,
            )
        self.use_shared_expert_gate = config.moe_shared_expert_gate
        # Keep bf16 activation for backward; the gate already holds the
        # high-precision copy.
        self.up_gate_proj.save_original_input = True
        if self.use_shared_expert_gate:
            self.gate_weight = paddle.create_parameter(
                shape=[config.hidden_size, 1],
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Initialize with Normal distribution aligned with Megatron.
            config.init_method(self.gate_weight)
        else:
            self.gate_weight = None

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        _dd = os.environ.get("MODEL_REPRO_MOE_DOWNSTREAM_DUMP_DIR")
        if _dd:
            import paddle.distributed as _pdd
            import os as _os

            _rk = _pdd.get_rank() if _pdd.is_initialized() else 0
            _os.makedirs(_dd, exist_ok=True)
            _li = getattr(self, "_shared_layer_no", "-")
            hidden_states.detach().astype("float32").cpu().numpy().tofile(
                _os.path.join(_dd, f"paddle_sh_fc1_in_{_li}_r{_rk}.f32.bin")
            )
        output, output_bias = None, None
        if os.environ.get("MODEL_REPRO_MOE_SHARED_TN", "0") == "1":
            # E-139/E-115: shared fc1 (K=6144 -> N=2048) differs by 1 ULP between
            # paddle F.linear and torch linear (333/122880 at M=60), while the
            # TN-materialized matmul (matmul(x, w.t().contiguous(),
            # transpose_y=True)) is 0-diff in both compat and default modes.
            # fc2 (K2048->N6144) is already 0-diff, so only fc1 is re-expressed;
            # the row-parallel down_proj (with its TP reduce) is left intact.
            _w1 = self.up_gate_proj.weight  # [in, N*2]
            fc1 = paddle.matmul(hidden_states, _w1.t().contiguous(), transpose_y=True)
            if self.up_gate_proj.bias is not None:
                fc1 = fc1 + self.up_gate_proj.bias
            _y1, _y2 = paddle.chunk(fc1, 2, axis=-1)
            act = F.silu(_y1) * _y2
            output, output_bias = self.down_proj(act)
        else:
            output, output_bias = super().forward(hidden_states)
        if _dd:
            output.detach().astype("float32").cpu().numpy().tofile(
                _os.path.join(
                    _dd, f"paddle_sh_post_fc2_{getattr(self, '_shared_layer_no', '-')}_r{_rk}.f32.bin"
                )
            )
        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score
        return output, output_bias
