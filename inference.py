"""
Traffic Intelligence Inference Pipeline
=======================================

Provides a high-level inference interface for running predictions on single
vehicle trajectory sequences or batched sequences using the trained
TCN-Transformer Gated Hybrid model.
"""

import os
import sys
from typing import Dict, List, Union, Any, Optional

import numpy as np
import torch
import joblib

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.tcn_transformer_hybrid import TCNTransformerHybrid


class TrafficInferencePipeline:
    """
    Production-ready inference pipeline for multi-task vehicle trajectory analytics.
    Handles feature scaling, model forward pass, and human-readable label decoding.
    """

    CLASS_LABELS = {
        "target_congestion_level": ["LOW", "MEDIUM", "HIGH"],
        "target_risk_level": ["SAFE", "WARNING", "HIGH"],
        "target_is_approaching": ["NON_APPROACHING", "APPROACHING"],
        "target_motion_state": ["STOPPED", "ACCELERATING", "DECELERATING", "CRUISING"],
        "target_maneuver_type": ["STATIONARY", "TURNING_LEFT", "TURNING_RIGHT", "STRAIGHT"],
        "target_has_infraction": ["COMPLIANT", "INFRACTION_DETECTED"],
        "target_infraction_type": ["NONE", "OVERSPEEDING", "ILLEGAL_STOPPING"]
    }

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pth",
        scaler_path: str = "checkpoints/feature_scaler.joblib",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

        # 1. Load fitted scaler
        self.scaler = joblib.load(scaler_path)

        # 2. Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        m_cfg = config.get("model", {})

        # 3. Instantiate architecture
        self.model = TCNTransformerHybrid(
            input_dim=config.get("dataset", {}).get("feature_count", 20),
            sequence_length=config.get("dataset", {}).get("sequence_length", 20),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            tcn_kernel_size=m_cfg.get("tcn", {}).get("kernel_size", 3),
            tcn_dilation_rates=m_cfg.get("tcn", {}).get("dilation_rates", [1, 2, 4, 8]),
            tcn_channels=m_cfg.get("tcn", {}).get("channels", [64, 64, 64, 64]),
            transformer_layers=m_cfg.get("transformer", {}).get("num_layers", 2),
            transformer_heads=m_cfg.get("transformer", {}).get("num_heads", 4),
            transformer_ff_dim=m_cfg.get("transformer", {}).get("feedforward_dim", 256),
            dropout=0.0  # Inference mode
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.epoch = checkpoint.get("epoch", None)
        self.val_loss = checkpoint.get("val_loss", None)

    def preprocess(self, sequence: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Applies fitted StandardScaler across sequence timesteps."""
        if isinstance(sequence, torch.Tensor):
            arr = sequence.cpu().numpy()
        else:
            arr = np.array(sequence, dtype=np.float32)

        # Ensure 3D shape [B, T, F]
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)

        b, t, f = arr.shape
        flat = arr.reshape(-1, f)
        scaled_flat = self.scaler.transform(flat)
        scaled_arr = scaled_flat.reshape(b, t, f).astype(np.float32)

        return torch.from_numpy(scaled_arr).to(self.device)

    def predict(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        return_raw_tensors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on single sequence [20, 20] or batch [B, 20, 20].
        Returns list of structured prediction dictionaries (one per sample).
        """
        tensor_x = self.preprocess(sequence)
        b = tensor_x.size(0)

        with torch.no_grad():
            raw_outputs = self.model(tensor_x)

        if return_raw_tensors:
            return raw_outputs

        results = []
        for i in range(b):
            sample_pred = {}

            # 1. Congestion Level
            cong_logits = raw_outputs["target_congestion_level"][i]
            cong_probs = torch.softmax(cong_logits, dim=-1).cpu().numpy()
            cong_class_idx = int(np.argmax(cong_probs))
            sample_pred["congestion_level"] = {
                "class_id": cong_class_idx,
                "label": self.CLASS_LABELS["target_congestion_level"][cong_class_idx],
                "confidence": float(cong_probs[cong_class_idx]),
                "probabilities": {lbl: float(p) for lbl, p in zip(self.CLASS_LABELS["target_congestion_level"], cong_probs)}
            }

            # 2. Congestion Score
            sample_pred["congestion_score"] = float(raw_outputs["target_congestion_score"][i].item())

            # 3. Risk Level
            risk_logits = raw_outputs["target_risk_level"][i]
            risk_probs = torch.softmax(risk_logits, dim=-1).cpu().numpy()
            risk_class_idx = int(np.argmax(risk_probs))
            sample_pred["risk_level"] = {
                "class_id": risk_class_idx,
                "label": self.CLASS_LABELS["target_risk_level"][risk_class_idx],
                "confidence": float(risk_probs[risk_class_idx]),
                "probabilities": {lbl: float(p) for lbl, p in zip(self.CLASS_LABELS["target_risk_level"], risk_probs)}
            }

            # 4. Approaching Status
            app_logits = raw_outputs["target_is_approaching"][i]
            app_probs = torch.softmax(app_logits, dim=-1).cpu().numpy()
            app_class_idx = int(np.argmax(app_probs))
            sample_pred["is_approaching"] = {
                "class_id": app_class_idx,
                "label": self.CLASS_LABELS["target_is_approaching"][app_class_idx],
                "confidence": float(app_probs[app_class_idx]),
                "is_approaching": bool(app_class_idx == 1)
            }

            # 5. Approach Threat Score
            sample_pred["approach_threat_score"] = float(raw_outputs["target_approach_score"][i].item())

            # 6. Motion State
            motion_logits = raw_outputs["target_motion_state"][i]
            motion_probs = torch.softmax(motion_logits, dim=-1).cpu().numpy()
            motion_class_idx = int(np.argmax(motion_probs))
            sample_pred["motion_state"] = {
                "class_id": motion_class_idx,
                "label": self.CLASS_LABELS["target_motion_state"][motion_class_idx],
                "confidence": float(motion_probs[motion_class_idx])
            }

            # 7. Maneuver Type
            man_logits = raw_outputs["target_maneuver_type"][i]
            man_probs = torch.softmax(man_logits, dim=-1).cpu().numpy()
            man_class_idx = int(np.argmax(man_probs))
            sample_pred["maneuver_type"] = {
                "class_id": man_class_idx,
                "label": self.CLASS_LABELS["target_maneuver_type"][man_class_idx],
                "confidence": float(man_probs[man_class_idx])
            }

            # 8. Infraction Detected
            inf_logits = raw_outputs["target_has_infraction"][i]
            inf_probs = torch.softmax(inf_logits, dim=-1).cpu().numpy()
            inf_class_idx = int(np.argmax(inf_probs))
            sample_pred["has_infraction"] = {
                "class_id": inf_class_idx,
                "label": self.CLASS_LABELS["target_has_infraction"][inf_class_idx],
                "confidence": float(inf_probs[inf_class_idx]),
                "violation_detected": bool(inf_class_idx == 1)
            }

            # 9. Infraction Type
            inft_logits = raw_outputs["target_infraction_type"][i]
            inft_probs = torch.softmax(inft_logits, dim=-1).cpu().numpy()
            inft_class_idx = int(np.argmax(inft_probs))
            sample_pred["infraction_type"] = {
                "class_id": inft_class_idx,
                "label": self.CLASS_LABELS["target_infraction_type"][inft_class_idx],
                "confidence": float(inft_probs[inft_class_idx])
            }

            # Auxiliary Next Displacement (dx, dy)
            if "aux_next_displacement" in raw_outputs:
                disp = raw_outputs["aux_next_displacement"][i].cpu().numpy()
                sample_pred["predicted_next_displacement"] = {
                    "dx": float(disp[0]),
                    "dy": float(disp[1])
                }

            results.append(sample_pred)

        return results
