"""
TCN-Transformer Gated Hybrid Architecture for Multi-Task Traffic Intelligence
=============================================================================

Proposed TCN-Transformer Gated Hybrid Architecture designed for multi-target
vehicle trajectory and traffic state prediction.

Components:
1. Input Layer: Flexible [Batch, Time, Features]
2. TCN Branch: Dilated causal 1D convolutions with residual connections
3. Transformer Branch: Multi-Head Self-Attention + Positional Encoding
4. Gated Fusion: Learnable element-wise gate balancing local vs global representations
5. Temporal Pooling: Attentive learnable sequence aggregator
6. Multi-Task Heads: Dedicated prediction heads for all 9 Phase 5 targets + auxiliary
7. Multi-Task Loss: Configurable static and uncertainty-weighted multi-objective loss
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. TEMPORAL CONVOLUTIONAL NETWORK (TCN) COMPONENTS
# ============================================================================

class ChausalConv1d(nn.Module):
    """
    1D Causal Convolution ensuring output at timestep t depends only on timesteps <= t.
    Achieved via left-padding of (kernel_size - 1) * dilation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, T]
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    A single residual block in the Temporal Convolutional Network.
    Contains two dilated causal convolutions, normalizations, nonlinear activations,
    dropouts, and a residual shortcut matching channel dimensions if needed.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Activation selector
        act_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        # First convolution branch
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.GroupNorm(1, out_channels)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(dropout)

        # Second convolution branch
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.GroupNorm(1, out_channels)
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(dropout)

        # Residual projection if in_channels != out_channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.out_act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, C, T]
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return self.out_act(out + residual)


class TCNBranch(nn.Module):
    """
    Temporal Convolutional Network branch for learning local/short-term temporal dynamics.
    Operates on [B, T, F] -> transposed to [B, F, T] for Conv1d -> returns [B, T, hidden_dim].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation_rates: Optional[List[int]] = None,
        channels: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        if channels is None:
            channels = [hidden_dim] * len(dilation_rates)

        assert len(dilation_rates) == len(channels), "Channels and dilation_rates must have same length"

        layers = []
        num_levels = len(dilation_rates)
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = dilation_rates[i]
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation=activation
                )
            )

        self.network = nn.Sequential(*layers)
        # Final projection to ensure output is exactly hidden_dim
        last_ch = channels[-1]
        if last_ch != hidden_dim:
            self.out_proj = nn.Conv1d(last_ch, hidden_dim, kernel_size=1)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        # Transpose to [B, F, T] for 1D convolutions
        x_trans = x.transpose(1, 2)
        out = self.network(x_trans)
        out = self.out_proj(out)
        # Transpose back to [B, T, hidden_dim]
        return out.transpose(1, 2)


# ============================================================================
# 2. TRANSFORMER ENCODER BRANCH COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding to inject temporal position information
    into the Transformer self-attention mechanism.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerBranch(nn.Module):
    """
    Transformer Encoder branch for capturing long-range temporal dependencies and global context.
    Includes input linear projection, positional encoding, multi-head self-attention,
    feed-forward networks, pre-layer normalization, and dropout.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, F]
        proj = self.input_proj(x)
        proj = self.pos_encoder(proj)
        out = self.transformer_encoder(proj, mask=mask)
        return self.norm(out)  # [B, T, hidden_dim]


# ============================================================================
# 3. GATED FUSION MECHANISM
# ============================================================================

class GatedFusion(nn.Module):
    """
    Learnable Gated Fusion mechanism between TCN representation Z_tcn and
    Transformer representation Z_trans.

    gate = sigmoid(W[Z_tcn ; Z_trans] + b)
    Z_fused = gate * Z_tcn + (1 - gate) * Z_trans

    This element-wise gating allows the model to dynamically choose between
    local high-frequency dynamics (TCN) and global context (Transformer) per feature and timestep.
    """
    def __init__(self, hidden_dim: int, gate_activation: str = "sigmoid"):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Concatenated representation has dimension 2 * hidden_dim
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid() if gate_activation == "sigmoid" else nn.Softmax(dim=-1)
        )

    def forward(self, z_tcn: torch.Tensor, z_trans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Both z_tcn and z_trans: [B, T, hidden_dim]
        assert z_tcn.shape == z_trans.shape, f"Shape mismatch: {z_tcn.shape} vs {z_trans.shape}"

        concat = torch.cat([z_tcn, z_trans], dim=-1)  # [B, T, 2 * hidden_dim]
        gate = self.gate_fc(concat)                    # [B, T, hidden_dim]

        z_fused = gate * z_tcn + (1.0 - gate) * z_trans
        return z_fused, gate


# ============================================================================
# 4. TEMPORAL ATTENTION POOLING
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Learnable attention-based temporal pooling mechanism.
    Weights each timestep t adaptively rather than naively flattening or simple averaging.

    a_t = softmax(w^T tanh(W * z_t + b))
    z_pooled = sum_t (a_t * z_t)
    """
    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        super().__init__()
        if attention_dim is None:
            attention_dim = max(16, hidden_dim // 2)

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, hidden_dim]
        scores = self.attention_net(x)               # [B, T, 1]
        weights = F.softmax(scores, dim=1)           # [B, T, 1]
        pooled = torch.sum(weights * x, dim=1)       # [B, hidden_dim]
        return pooled, weights.squeeze(-1)           # [B, hidden_dim], [B, T]


# ============================================================================
# 5. MULTI-TASK OUTPUT HEADS
# ============================================================================

class TaskHead(nn.Module):
    """
    Individual task output projection head with non-linear intermediate MLP.
    Supports classification and regression targets.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        task_type: str,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.task_type = task_type
        self.out_dim = out_dim
        if mlp_dim is None:
            mlp_dim = max(32, in_dim // 2)

        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Squeeze single-output regressions to [B] for consistency
        if self.task_type == "regression" and self.out_dim == 1:
            out = out.squeeze(-1)
        return out


class MultiTaskOutputHeads(nn.Module):
    """
    Comprehensive multi-task projection heads supporting all 9 targets discovered
    in the Phase 5 project audit, plus an optional auxiliary trajectory displacement head.
    """
    def __init__(
        self,
        hidden_dim: int,
        target_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Default configuration derived from Phase 5 audit
        if target_configs is None:
            target_configs = {
                # 1. Congestion classification (LOW, MEDIUM, HIGH)
                "target_congestion_level": {"type": "multiclass", "dim": 3},
                # 2. Congestion score regression (10.0 to 100.0)
                "target_congestion_score": {"type": "regression", "dim": 1},
                # 3. Risk level classification (SAFE, WARNING, HIGH)
                "target_risk_level": {"type": "multiclass", "dim": 3},
                # 4. Approaching status binary classification (0 or 1)
                "target_is_approaching": {"type": "binary", "dim": 2},
                # 5. Approach threat score regression (0.03 to 100.0)
                "target_approach_score": {"type": "regression", "dim": 1},
                # 6. Motion state classification (STOPPED, ACCELERATING, DECELERATING, CRUISING)
                "target_motion_state": {"type": "multiclass", "dim": 4},
                # 7. Maneuver type classification (STATIONARY, TURNING_LEFT, TURNING_RIGHT, STRAIGHT)
                "target_maneuver_type": {"type": "multiclass", "dim": 4},
                # 8. Infraction detected binary classification (COMPLIANT, INFRACTION)
                "target_has_infraction": {"type": "binary", "dim": 2},
                # 9. Infraction type classification (NONE, OVERSPEEDING, ILLEGAL_STOPPING)
                "target_infraction_type": {"type": "multiclass", "dim": 3},
                # Auxiliary: Next-step trajectory displacement [dx, dy]
                "aux_next_displacement": {"type": "regression", "dim": 2}
            }

        self.target_configs = target_configs
        self.heads = nn.ModuleDict()

        for target_name, cfg in target_configs.items():
            t_type = cfg.get("type", "multiclass")
            out_dim = cfg.get("dim", 1)
            self.heads[target_name] = TaskHead(
                in_dim=hidden_dim,
                out_dim=out_dim,
                task_type=t_type,
                dropout=dropout
            )

    def forward(self, pooled_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        for target_name, head in self.heads.items():
            outputs[target_name] = head(pooled_features)
        return outputs


# ============================================================================
# 6. MASTER HYBRID ARCHITECTURE
# ============================================================================

class TCNTransformerHybrid(nn.Module):
    """
    Proposed TCN-Transformer Gated Hybrid Architecture for Traffic Intelligence.

    Combines:
    - Input projection & feature encoding
    - TCN branch for local dynamics
    - Transformer branch for long-range context
    - Learnable gated fusion
    - Attentive temporal pooling
    - Multi-task output heads
    """
    def __init__(
        self,
        input_dim: int = 20,
        sequence_length: int = 20,
        hidden_dim: int = 64,
        tcn_kernel_size: int = 3,
        tcn_dilation_rates: Optional[List[int]] = None,
        tcn_channels: Optional[List[int]] = None,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 256,
        dropout: float = 0.2,
        target_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        if tcn_dilation_rates is None:
            tcn_dilation_rates = [1, 2, 4, 8]
        if tcn_channels is None:
            tcn_channels = [hidden_dim] * len(tcn_dilation_rates)

        # 1. Temporal Convolutional Branch
        self.tcn_branch = TCNBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=tcn_kernel_size,
            dilation_rates=tcn_dilation_rates,
            channels=tcn_channels,
            dropout=dropout
        )

        # 2. Transformer Encoder Branch
        self.transformer_branch = TransformerBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            feedforward_dim=transformer_ff_dim,
            dropout=dropout
        )

        # 3. Gated Fusion Mechanism
        self.gated_fusion = GatedFusion(hidden_dim=hidden_dim)

        # 4. Attentive Temporal Pooling
        self.temporal_pooling = AttentionPooling(hidden_dim=hidden_dim)

        # 5. Multi-Task Output Heads
        self.output_heads = MultiTaskOutputHeads(
            hidden_dim=hidden_dim,
            target_configs=target_configs,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape [Batch, Sequence_Length, Features]
            return_intermediates: If True, returns gating weights and attention maps.
        Returns:
            Dictionary of predictions across all target heads.
        """
        assert x.ndim == 3, f"Expected 3D input tensor [B, T, F], got shape {x.shape}"
        assert x.size(-1) == self.input_dim, f"Feature dimension mismatch: expected {self.input_dim}, got {x.size(-1)}"

        # 1. Branch computation
        z_tcn = self.tcn_branch(x)                    # [B, T, hidden_dim]
        z_trans = self.transformer_branch(x)          # [B, T, hidden_dim]

        # 2. Gated fusion
        z_fused, gate = self.gated_fusion(z_tcn, z_trans)  # [B, T, hidden_dim], [B, T, hidden_dim]

        # 3. Temporal pooling
        z_pooled, attn_weights = self.temporal_pooling(z_fused)  # [B, hidden_dim], [B, T]

        # 4. Multi-task output heads
        outputs = self.output_heads(z_pooled)

        if return_intermediates:
            intermediates = {
                "z_tcn": z_tcn,
                "z_trans": z_trans,
                "z_fused": z_fused,
                "gate": gate,
                "z_pooled": z_pooled,
                "attention_weights": attn_weights
            }
            return outputs, intermediates

        return outputs


# ============================================================================
# 7. MULTI-TASK LOSS FUNCTION
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Configurable multi-task loss supporting:
    - Static weighted loss: L_total = sum(w_i * L_i)
    - Uncertainty-based task weighting (Kendall & Gal, 2018):
        L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
      where s_i = log(sigma_i^2) is a learnable parameter.
    - Automatic selection of loss functions:
        * multiclass -> CrossEntropyLoss (with optional class weights)
        * binary -> CrossEntropyLoss or BCEWithLogitsLoss
        * regression -> SmoothL1Loss (Huber) or MSELoss
    """
    def __init__(
        self,
        task_configs: Dict[str, Dict[str, Any]],
        weighting_mode: str = "static",
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        self.task_configs = task_configs
        self.weighting_mode = weighting_mode.lower()
        self.task_names = list(task_configs.keys())

        # Static loss weights
        self.static_weights = {}
        for name, cfg in task_configs.items():
            self.static_weights[name] = float(cfg.get("weight", 1.0))

        # Learnable log variances for Kendall & Gal uncertainty weighting
        if self.weighting_mode == "uncertainty":
            self.log_vars = nn.ParameterDict()
            for name in self.task_names:
                self.log_vars[name] = nn.Parameter(torch.zeros(1))

        # Individual loss criteria
        self.loss_criteria = nn.ModuleDict()
        for name, cfg in task_configs.items():
            t_type = cfg.get("type", "multiclass")
            loss_name = cfg.get("loss", "auto")

            if t_type == "multiclass" or t_type == "binary":
                # Check for class weights
                cw = class_weights.get(name) if class_weights else None
                self.loss_criteria[name] = nn.CrossEntropyLoss(weight=cw)
            elif t_type == "regression":
                if loss_name == "mse":
                    self.loss_criteria[name] = nn.MSELoss()
                else:
                    self.loss_criteria[name] = nn.SmoothL1Loss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the weighted multi-task loss.
        """
        individual_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for name in self.task_names:
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]
            criterion = self.loss_criteria[name]
            t_type = self.task_configs[name].get("type", "multiclass")

            if t_type in ("multiclass", "binary"):
                # Target must be long tensor
                loss_i = criterion(pred, target.long())
            else:
                # Regression target must be float tensor
                loss_i = criterion(pred.float(), target.float())

            individual_losses[name] = loss_i

            if self.weighting_mode == "uncertainty":
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                task_loss = 0.5 * precision * loss_i + 0.5 * log_var
                total_loss = total_loss + task_loss.squeeze()
            else:
                w = self.static_weights.get(name, 1.0)
                total_loss = total_loss + w * loss_i

        individual_losses["total_loss"] = total_loss
        return total_loss, individual_losses
