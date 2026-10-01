"""
Real Batch Validation Test for TCN-Transformer Gated Hybrid
===========================================================

Verifies forward and backward pass on an actual batch loaded from
data/hybrid_traffic/sequences/train_sequences.npz and
data/hybrid_traffic/labels/train_labels.npz in STRICT READ-ONLY mode.
"""

import sys
import os
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.tcn_transformer_hybrid import TCNTransformerHybrid, MultiTaskLoss


class TestRealDatasetBatch(unittest.TestCase):
    """Verifies model against actual Phase 4 training tensors in read-only mode."""

    def test_real_training_batch(self):
        seq_path = "data/hybrid_traffic/sequences/train_sequences.npz"
        lbl_path = "data/hybrid_traffic/labels/train_labels.npz"

        self.assertTrue(os.path.exists(seq_path), f"Missing {seq_path}")
        self.assertTrue(os.path.exists(lbl_path), f"Missing {lbl_path}")

        # Load real data strictly in read-only mode (mmap_mode='r')
        with np.load(seq_path, mmap_mode='r') as seq_data:
            X_all = seq_data["X"]
            y_disp_all = seq_data["y_next_disp"]
            # Extract first batch of 16 sequences
            batch_X = torch.from_numpy(np.array(X_all[:16], dtype=np.float32))
            batch_disp = torch.from_numpy(np.array(y_disp_all[:16], dtype=np.float32))

        with np.load(lbl_path, mmap_mode='r') as lbl_data:
            batch_targets = {
                "target_congestion_level": torch.from_numpy(np.array(lbl_data["target_congestion_level"][:16], dtype=np.int64)),
                "target_congestion_score": torch.from_numpy(np.array(lbl_data["target_congestion_score"][:16], dtype=np.float32)),
                "target_risk_level": torch.from_numpy(np.array(lbl_data["target_risk_level"][:16], dtype=np.int64)),
                "target_is_approaching": torch.from_numpy(np.array(lbl_data["target_is_approaching"][:16], dtype=np.int64)),
                "target_approach_score": torch.from_numpy(np.array(lbl_data["target_approach_score"][:16], dtype=np.float32)),
                "target_motion_state": torch.from_numpy(np.array(lbl_data["target_motion_state"][:16], dtype=np.int64)),
                "target_maneuver_type": torch.from_numpy(np.array(lbl_data["target_maneuver_type"][:16], dtype=np.int64)),
                "target_has_infraction": torch.from_numpy(np.array(lbl_data["target_has_infraction"][:16], dtype=np.int64)),
                "target_infraction_type": torch.from_numpy(np.array(lbl_data["target_infraction_type"][:16], dtype=np.int64)),
                "aux_next_displacement": batch_disp
            }

        self.assertEqual(batch_X.shape, (16, 20, 20))

        # Instantiate model
        model = TCNTransformerHybrid(
            input_dim=20,
            sequence_length=20,
            hidden_dim=64,
            tcn_kernel_size=3,
            tcn_dilation_rates=[1, 2, 4, 8],
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff_dim=256,
            dropout=0.1
        )
        model.train()

        # Forward pass on real batch
        predictions = model(batch_X)
        self.assertEqual(len(predictions), 10)

        # Multi-task loss on real batch
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
        total_loss, ind_losses = criterion(predictions, batch_targets)

        self.assertTrue(torch.isfinite(total_loss).item())
        self.assertGreater(total_loss.item(), 0.0)

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"Parameter {name} has NaN gradient")

        print(f"\n[REAL BATCH VERIFICATION PASSED] Loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    unittest.main()
