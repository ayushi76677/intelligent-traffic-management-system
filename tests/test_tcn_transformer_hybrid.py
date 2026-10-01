"""
Unit Tests for Proposed TCN-Transformer Gated Hybrid Architecture
=================================================================

Tests:
1. Model imports correctly
2. Forward pass runs with correct tensor dimensions
3. Every target head produces valid output
4. No NaNs or Infs appear in model outputs
5. Backward pass computes valid gradients across all branches
6. Loss calculation functions properly (static and uncertainty weighting)
7. CPU inference succeeds
8. GPU inference succeeds if CUDA is available
9. Gated fusion values lie strictly in [0, 1]
10. Attentive pooling weights are valid probability distributions
11. Model generalizes across variable batch sizes and sequence lengths
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.tcn_transformer_hybrid import (
    TemporalBlock,
    TCNBranch,
    PositionalEncoding,
    TransformerBranch,
    GatedFusion,
    AttentionPooling,
    TaskHead,
    MultiTaskOutputHeads,
    TCNTransformerHybrid,
    MultiTaskLoss
)


class TestTCNTransformerHybrid(unittest.TestCase):
    """Test suite for TCN-Transformer Gated Hybrid model."""

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 20
        self.input_dim = 20
        self.hidden_dim = 64

        self.model = TCNTransformerHybrid(
            input_dim=self.input_dim,
            sequence_length=self.seq_len,
            hidden_dim=self.hidden_dim,
            tcn_kernel_size=3,
            tcn_dilation_rates=[1, 2, 4, 8],
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff_dim=128,
            dropout=0.1
        )

        # Expected target specifications from Phase 5 audit
        self.target_specs = {
            "target_congestion_level": {"type": "multiclass", "shape": (self.batch_size, 3)},
            "target_congestion_score": {"type": "regression", "shape": (self.batch_size,)},
            "target_risk_level": {"type": "multiclass", "shape": (self.batch_size, 3)},
            "target_is_approaching": {"type": "binary", "shape": (self.batch_size, 2)},
            "target_approach_score": {"type": "regression", "shape": (self.batch_size,)},
            "target_motion_state": {"type": "multiclass", "shape": (self.batch_size, 4)},
            "target_maneuver_type": {"type": "multiclass", "shape": (self.batch_size, 4)},
            "target_has_infraction": {"type": "binary", "shape": (self.batch_size, 2)},
            "target_infraction_type": {"type": "multiclass", "shape": (self.batch_size, 3)},
            "aux_next_displacement": {"type": "regression", "shape": (self.batch_size, 2)}
        }

    def test_01_imports(self):
        """Verify all essential classes import and instantiate properly."""
        self.assertIsNotNone(TemporalBlock)
        self.assertIsNotNone(TCNBranch)
        self.assertIsNotNone(TransformerBranch)
        self.assertIsNotNone(GatedFusion)
        self.assertIsNotNone(AttentionPooling)
        self.assertIsNotNone(MultiTaskOutputHeads)
        self.assertIsNotNone(TCNTransformerHybrid)
        self.assertIsNotNone(MultiTaskLoss)

    def test_02_forward_pass_and_dimensions(self):
        """Verify forward pass executes and tensor dimensions match specifications."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        outputs = self.model(x)

        self.assertIsInstance(outputs, dict)
        for target_name, spec in self.target_specs.items():
            self.assertIn(target_name, outputs, f"Missing output head for {target_name}")
            out_tensor = outputs[target_name]
            expected_shape = spec["shape"]
            self.assertEqual(
                out_tensor.shape,
                expected_shape,
                f"Shape mismatch for {target_name}: expected {expected_shape}, got {out_tensor.shape}"
            )

    def test_03_no_nan_or_inf(self):
        """Verify outputs contain zero NaN and zero Inf values."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        outputs = self.model(x)

        for target_name, out_tensor in outputs.items():
            self.assertFalse(
                torch.isnan(out_tensor).any().item(),
                f"NaN detected in {target_name} output"
            )
            self.assertFalse(
                torch.isinf(out_tensor).any().item(),
                f"Inf detected in {target_name} output"
            )

    def test_04_backward_pass_and_gradients(self):
        """Verify loss computation and backward pass compute valid gradients."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim, requires_grad=False)
        self.model.train()
        outputs = self.model(x)

        # Construct dummy targets matching Phase 5 data types
        targets = {
            "target_congestion_level": torch.randint(0, 3, (self.batch_size,)),
            "target_congestion_score": torch.rand(self.batch_size) * 90.0 + 10.0,
            "target_risk_level": torch.randint(0, 3, (self.batch_size,)),
            "target_is_approaching": torch.randint(0, 2, (self.batch_size,)),
            "target_approach_score": torch.rand(self.batch_size) * 99.0 + 0.03,
            "target_motion_state": torch.randint(0, 4, (self.batch_size,)),
            "target_maneuver_type": torch.randint(0, 4, (self.batch_size,)),
            "target_has_infraction": torch.randint(0, 2, (self.batch_size,)),
            "target_infraction_type": torch.randint(0, 3, (self.batch_size,)),
            "aux_next_displacement": torch.randn(self.batch_size, 2)
        }

        task_configs = {
            "target_congestion_level": {"type": "multiclass", "weight": 1.0},
            "target_congestion_score": {"type": "regression", "weight": 0.5},
            "target_risk_level": {"type": "multiclass", "weight": 1.2},
            "target_is_approaching": {"type": "binary", "weight": 1.0},
            "target_approach_score": {"type": "regression", "weight": 0.5},
            "target_motion_state": {"type": "multiclass", "weight": 1.0},
            "target_maneuver_type": {"type": "multiclass", "weight": 1.0},
            "target_has_infraction": {"type": "binary", "weight": 1.2},
            "target_infraction_type": {"type": "multiclass", "weight": 0.8},
            "aux_next_displacement": {"type": "regression", "weight": 0.5}
        }

        criterion = MultiTaskLoss(task_configs=task_configs, weighting_mode="static")
        total_loss, ind_losses = criterion(outputs, targets)

        self.assertTrue(torch.isfinite(total_loss).item(), "Total loss is not finite")
        self.assertGreater(total_loss.item(), 0.0, "Total loss must be positive")

        # Zero gradients & backprop
        self.model.zero_grad()
        total_loss.backward()

        # Check gradients in key branches
        has_grad_tcn = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.tcn_branch.parameters())
        has_grad_trans = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.transformer_branch.parameters())
        has_grad_gate = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.gated_fusion.parameters())
        has_grad_pool = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.temporal_pooling.parameters())

        self.assertTrue(has_grad_tcn, "TCN branch did not receive gradients")
        self.assertTrue(has_grad_trans, "Transformer branch did not receive gradients")
        self.assertTrue(has_grad_gate, "Gated fusion did not receive gradients")
        self.assertTrue(has_grad_pool, "Temporal pooling did not receive gradients")

    def test_05_gated_fusion_and_pooling_properties(self):
        """Verify gating values are in [0, 1] and pooling attention weights sum to 1."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        outputs, intermediates = self.model(x, return_intermediates=True)

        gate = intermediates["gate"]
        self.assertEqual(gate.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertTrue((gate >= 0.0).all().item() and (gate <= 1.0).all().item(), "Gate values must be in [0, 1]")

        attn_weights = intermediates["attention_weights"]
        self.assertEqual(attn_weights.shape, (self.batch_size, self.seq_len))
        sums = attn_weights.sum(dim=-1)
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
            "Temporal pooling weights must sum to 1.0 across sequence length"
        )

    def test_06_uncertainty_loss_weighting(self):
        """Verify Kendall & Gal uncertainty loss weighting adapts log-variances."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        outputs = self.model(x)

        targets = {
            "target_congestion_level": torch.randint(0, 3, (self.batch_size,)),
            "target_congestion_score": torch.rand(self.batch_size) * 90.0 + 10.0,
            "target_risk_level": torch.randint(0, 3, (self.batch_size,)),
            "target_is_approaching": torch.randint(0, 2, (self.batch_size,)),
            "target_approach_score": torch.rand(self.batch_size) * 99.0 + 0.03,
            "target_motion_state": torch.randint(0, 4, (self.batch_size,)),
            "target_maneuver_type": torch.randint(0, 4, (self.batch_size,)),
            "target_has_infraction": torch.randint(0, 2, (self.batch_size,)),
            "target_infraction_type": torch.randint(0, 3, (self.batch_size,)),
            "aux_next_displacement": torch.randn(self.batch_size, 2)
        }

        task_configs = {k: {"type": self.target_specs[k]["type"]} for k in self.target_specs}
        criterion = MultiTaskLoss(task_configs=task_configs, weighting_mode="uncertainty")

        loss, _ = criterion(outputs, targets)
        loss.backward()

        for name, log_var in criterion.log_vars.items():
            self.assertIsNotNone(log_var.grad, f"Log-variance for {name} did not receive gradients")

    def test_07_device_execution(self):
        """Verify CPU inference and GPU inference if CUDA is available."""
        # 1. CPU execution
        cpu_device = torch.device("cpu")
        self.model.to(cpu_device)
        self.model.eval()
        with torch.no_grad():
            x_cpu = torch.randn(2, self.seq_len, self.input_dim, device=cpu_device)
            out_cpu = self.model(x_cpu)
            self.assertIn("target_congestion_level", out_cpu)

        # 2. CUDA execution if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            self.model.to(cuda_device)
            with torch.no_grad():
                x_gpu = torch.randn(2, self.seq_len, self.input_dim, device=cuda_device)
                out_gpu = self.model(x_gpu)
                self.assertIn("target_congestion_level", out_gpu)
            self.model.to(cpu_device)
        else:
            print("\n[NOTE] CUDA is not available on this system (AMD CPU/iGPU). Tested CPU execution successfully.")

    def test_08_flexible_dimensions(self):
        """Verify model handles different sequence lengths and batch sizes gracefully."""
        # Batch size 1, sequence length 15
        x1 = torch.randn(1, 15, self.input_dim)
        out1 = self.model(x1)
        self.assertEqual(out1["target_congestion_level"].shape, (1, 3))

        # Batch size 8, sequence length 30
        x2 = torch.randn(8, 30, self.input_dim)
        out2 = self.model(x2)
        self.assertEqual(out2["target_congestion_level"].shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
