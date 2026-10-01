"""
Real-Time Hybrid Model Inference Pipeline
=========================================

Production-grade real-time inference interface that integrates:
1. Live track observations via TrackSequenceBuffer
2. Fitted StandardScaler from Phase 4/5 training (checkpoints/feature_scaler.joblib)
3. Frozen TCN-Transformer Gated Hybrid model (checkpoints/best_model.pth)
4. Multi-head output decoding with confidence scoring and uncertainty gates

Adheres to:
- Exact sequence length = 20
- Exact feature dimension = 20
- Ground-truth target names from checkpoints/best_model_config.json
- Zero retraining or alteration of weights/scalers
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import joblib

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.tcn_transformer_hybrid import TCNTransformerHybrid
from ml.hybrid_traffic.feature_adapter import LiveFeatureAdapter, EXACT_FEATURE_NAMES, FEATURE_COUNT
from ml.hybrid_traffic.sequence_buffer import TrackSequenceBuffer, TrackBufferState

logger = logging.getLogger("TrafficIntelligence.RealtimeInference")


class RealTimeTrafficPredictor:
    """
    Core production inference engine for the TCN-Transformer Gated Hybrid model.
    Maintains per-track temporal sequences, handles feature scaling, model inference,
    multi-task decoding, and confidence/uncertainty evaluation.
    """

    CLASS_LABELS = {
        "target_congestion_level": ["LOW", "MEDIUM", "HIGH"],
        "target_risk_level": ["SAFE", "WARNING", "HIGH"],
        "target_is_approaching": ["NON_APPROACHING", "APPROACHING"],
        "target_motion_state": ["STOPPED", "ACCELERATING", "DECELERATING", "CRUISING"],
        "target_maneuver_type": ["STATIONARY", "TURNING_LEFT", "TURNING_RIGHT", "STRAIGHT"],
        "target_has_infraction": ["COMPLIANT", "INFRACTION"],
        "target_infraction_type": ["NONE", "OVERSPEEDING", "ILLEGAL_STOPPING"],
    }

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pth",
        scaler_path: str = "checkpoints/feature_scaler.joblib",
        config_path: str = "checkpoints/best_model_config.json",
        device: Optional[str] = None,
        confidence_threshold: float = 0.60,
        max_idle_frames: int = 30,
        feature_adapter: Optional[LiveFeatureAdapter] = None,
    ):
        self.checkpoint_path = os.path.join(BASE_DIR, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
        self.scaler_path = os.path.join(BASE_DIR, scaler_path) if not os.path.isabs(scaler_path) else scaler_path
        self.config_path = os.path.join(BASE_DIR, config_path) if not os.path.isabs(config_path) else config_path

        self.confidence_threshold = confidence_threshold
        self.sequence_length = 20
        self.feature_count = 20

        # Select compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load fitted scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Feature scaler not found at: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"[RealTimeInference] Loaded feature scaler from {self.scaler_path}")

        # 2. Load model config
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)
        else:
            self.model_config = {}

        # 3. Instantiate model architecture and load frozen weights
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = checkpoint.get("config", self.model_config.get("config", {}))
        m_cfg = cfg.get("model", {})
        d_cfg = cfg.get("dataset", {})

        self.input_dim = d_cfg.get("feature_count", 20)
        self.sequence_length = d_cfg.get("sequence_length", 20)

        self.model = TCNTransformerHybrid(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=m_cfg.get("hidden_dim", 64),
            tcn_kernel_size=m_cfg.get("tcn", {}).get("kernel_size", 3),
            tcn_dilation_rates=m_cfg.get("tcn", {}).get("dilation_rates", [1, 2, 4, 8]),
            tcn_channels=m_cfg.get("tcn", {}).get("channels", [64, 64, 64, 64]),
            transformer_layers=m_cfg.get("transformer", {}).get("num_layers", 2),
            transformer_heads=m_cfg.get("transformer", {}).get("num_heads", 4),
            transformer_ff_dim=m_cfg.get("transformer", {}).get("feedforward_dim", 256),
            dropout=0.0  # Production inference mode
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.epoch = checkpoint.get("epoch", None)
        self.val_loss = checkpoint.get("val_loss", None)
        logger.info(
            f"[RealTimeInference] TCN-Transformer Gated Hybrid model loaded successfully. "
            f"Device: {self.device}, Epoch: {self.epoch}, Val Loss: {self.val_loss}"
        )

        # 4. Initialize per-track temporal sequence buffer
        self.feature_adapter = feature_adapter or LiveFeatureAdapter()
        self.buffer = TrackSequenceBuffer(
            sequence_length=self.sequence_length,
            max_idle_frames=max_idle_frames,
            feature_adapter=self.feature_adapter,
        )

        # Performance telemetry
        self.telemetry = {
            "total_inferences": 0,
            "total_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
            "last_latency_ms": 0.0,
        }

    def preprocess_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Applies fitted StandardScaler on sequence array of shape (20, 20) or batch (B, 20, 20).
        """
        arr = np.array(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # (1, 20, 20)

        b, t, f = arr.shape
        flat = arr.reshape(-1, f)
        # Handle any possible non-finite entries before transform
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_flat = self.scaler.transform(flat)
        scaled_arr = scaled_flat.reshape(b, t, f).astype(np.float32)

        return torch.from_numpy(scaled_arr).to(self.device)

    def _decode_predictions(self, raw_outputs: Dict[str, torch.Tensor], idx: int = 0) -> Tuple[Dict[str, Any], float, bool]:
        """
        Decodes raw model logits and regression heads into structured predictions.
        Computes aggregated confidence score and flags safety uncertainty.
        """
        preds = {}
        confidences = []

        # 1. Congestion Level (Multiclass: LOW, MEDIUM, HIGH)
        cong_logits = raw_outputs["target_congestion_level"][idx]
        cong_probs = torch.softmax(cong_logits, dim=-1).cpu().numpy()
        cong_idx = int(np.argmax(cong_probs))
        cong_conf = float(cong_probs[cong_idx])
        confidences.append(cong_conf)
        preds["congestion_level"] = {
            "class_id": cong_idx,
            "label": self.CLASS_LABELS["target_congestion_level"][cong_idx],
            "confidence": round(cong_conf, 4),
            "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(self.CLASS_LABELS["target_congestion_level"], cong_probs)},
        }

        # 2. Congestion Score (Regression: [10.0, 100.0])
        raw_cong_score = float(raw_outputs["target_congestion_score"][idx].item())
        clamped_cong_score = max(10.0, min(100.0, raw_cong_score))
        preds["congestion_score"] = round(clamped_cong_score, 2)

        # 3. Risk Level (Multiclass: SAFE, WARNING, HIGH)
        risk_logits = raw_outputs["target_risk_level"][idx]
        risk_probs = torch.softmax(risk_logits, dim=-1).cpu().numpy()
        risk_idx = int(np.argmax(risk_probs))
        risk_conf = float(risk_probs[risk_idx])
        confidences.append(risk_conf)
        preds["risk_level"] = {
            "class_id": risk_idx,
            "label": self.CLASS_LABELS["target_risk_level"][risk_idx],
            "confidence": round(risk_conf, 4),
            "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(self.CLASS_LABELS["target_risk_level"], risk_probs)},
        }

        # 4. Approaching Status (Binary: NON_APPROACHING, APPROACHING)
        app_logits = raw_outputs["target_is_approaching"][idx]
        app_probs = torch.softmax(app_logits, dim=-1).cpu().numpy()
        app_idx = int(np.argmax(app_probs))
        app_conf = float(app_probs[app_idx])
        confidences.append(app_conf)
        preds["is_approaching"] = {
            "class_id": app_idx,
            "label": self.CLASS_LABELS["target_is_approaching"][app_idx],
            "confidence": round(app_conf, 4),
            "is_approaching": bool(app_idx == 1),
        }

        # 5. Approach Threat Score (Regression: [0.03, 100.0])
        raw_app_score = float(raw_outputs["target_approach_score"][idx].item())
        clamped_app_score = max(0.0, min(100.0, raw_app_score))
        preds["approach_threat_score"] = round(clamped_app_score, 2)

        # 6. Motion State (Multiclass: STOPPED, ACCELERATING, DECELERATING, CRUISING)
        motion_logits = raw_outputs["target_motion_state"][idx]
        motion_probs = torch.softmax(motion_logits, dim=-1).cpu().numpy()
        motion_idx = int(np.argmax(motion_probs))
        motion_conf = float(motion_probs[motion_idx])
        confidences.append(motion_conf)
        preds["motion_state"] = {
            "class_id": motion_idx,
            "label": self.CLASS_LABELS["target_motion_state"][motion_idx],
            "confidence": round(motion_conf, 4),
        }

        # 7. Maneuver Type (Multiclass: STATIONARY, TURNING_LEFT, TURNING_RIGHT, STRAIGHT)
        man_logits = raw_outputs["target_maneuver_type"][idx]
        man_probs = torch.softmax(man_logits, dim=-1).cpu().numpy()
        man_idx = int(np.argmax(man_probs))
        man_conf = float(man_probs[man_idx])
        confidences.append(man_conf)
        preds["maneuver_type"] = {
            "class_id": man_idx,
            "label": self.CLASS_LABELS["target_maneuver_type"][man_idx],
            "confidence": round(man_conf, 4),
        }

        # 8. Infraction Detected (Binary: COMPLIANT, INFRACTION)
        inf_logits = raw_outputs["target_has_infraction"][idx]
        inf_probs = torch.softmax(inf_logits, dim=-1).cpu().numpy()
        inf_idx = int(np.argmax(inf_probs))
        inf_conf = float(inf_probs[inf_idx])
        confidences.append(inf_conf)
        preds["has_infraction"] = {
            "class_id": inf_idx,
            "label": self.CLASS_LABELS["target_has_infraction"][inf_idx],
            "confidence": round(inf_conf, 4),
            "violation_detected": bool(inf_idx == 1),
        }

        # 9. Infraction Type (Multiclass: NONE, OVERSPEEDING, ILLEGAL_STOPPING)
        inft_logits = raw_outputs["target_infraction_type"][idx]
        inft_probs = torch.softmax(inft_logits, dim=-1).cpu().numpy()
        inft_idx = int(np.argmax(inft_probs))
        inft_conf = float(inft_probs[inft_idx])
        confidences.append(inft_conf)
        preds["infraction_type"] = {
            "class_id": inft_idx,
            "label": self.CLASS_LABELS["target_infraction_type"][inft_idx],
            "confidence": round(inft_conf, 4),
        }

        # 10. Auxiliary Next Displacement (Regression: dx, dy)
        if "aux_next_displacement" in raw_outputs:
            disp = raw_outputs["aux_next_displacement"][idx].cpu().numpy()
            preds["predicted_next_displacement"] = {
                "dx": round(float(disp[0]), 2),
                "dy": round(float(disp[1]), 2),
            }

        # Aggregate overall prediction confidence (mean across classification heads)
        overall_confidence = float(np.mean(confidences))
        is_uncertain = bool(overall_confidence < self.confidence_threshold)

        return preds, round(overall_confidence, 4), is_uncertain

    def predict_live(self, track_observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary interface: Receives single live track observation, updates buffer,
        runs model if ready, and returns structured predictions or warm-up status.
        
        Args:
            track_observation: Dict with keys 'track_id', 'x1', 'y1', 'x2', 'y2', optional 'frame', 'time'.
            
        Returns:
            Dict containing:
            - track_id
            - prediction_ready (bool)
            - status ("READY" or "WARMING_UP")
            - warmup_progress (float)
            - predictions (dict or None)
            - confidence (float)
            - is_uncertain (bool)
            - timestamp (float)
            - zone_id (str)
            - lane_id (str)
        """
        t0 = time.perf_counter()
        buffer_state = self.buffer.update(track_observation)
        obs_time = buffer_state.last_time

        # Check if sequence is warming up
        if not buffer_state.is_ready:
            return {
                "track_id": buffer_state.track_id,
                "prediction_ready": False,
                "status": "WARMING_UP",
                "warmup_progress": buffer_state.warmup_progress,
                "observations_recorded": buffer_state.observation_count,
                "observations_required": self.sequence_length,
                "predictions": None,
                "confidence": 0.0,
                "is_uncertain": True,
                "safety_advisory": "Trajectory warming up — collecting temporal history.",
                "timestamp": obs_time,
                "zone_id": buffer_state.zone_id,
                "lane_id": buffer_state.lane_id,
            }

        # Sequence is READY — execute inference
        try:
            tensor_x = self.preprocess_sequence(buffer_state.sequence)
            with torch.no_grad():
                raw_outputs = self.model(tensor_x)
            predictions, confidence, is_uncertain = self._decode_predictions(raw_outputs, idx=0)
            safety_advisory = self._generate_safety_advisory(predictions, is_uncertain)
        except Exception as e:
            logger.error(f"[RealTimeInference] Inference error on track {buffer_state.track_id}: {e}", exc_info=True)
            predictions = None
            confidence = 0.0
            is_uncertain = True
            safety_advisory = f"Inference degraded due to numerical anomaly: {e}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._update_telemetry(latency_ms)
        logger.debug(f"[RealTimeInference] Track {buffer_state.track_id} inference finished in {latency_ms:.2f} ms (conf: {confidence:.2f})")

        return {
            "track_id": buffer_state.track_id,
            "prediction_ready": True,
            "status": "READY",
            "warmup_progress": 1.0,
            "observations_recorded": buffer_state.observation_count,
            "observations_required": self.sequence_length,
            "predictions": predictions,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "safety_advisory": safety_advisory,
            "timestamp": obs_time,
            "zone_id": buffer_state.zone_id,
            "lane_id": buffer_state.lane_id,
            "latency_ms": round(latency_ms, 2),
        }

    def predict_frame_tracks(
        self,
        frame_num: int,
        track_observations: List[Dict[str, Any]],
        cleanup_idle: bool = True
    ) -> List[Dict[str, Any]]:
        """
        High-throughput batched inference for all tracks observed in a single video frame.
        Batches ready sequences into a single forward pass while returning warm-up states
        for immature tracks.
        """
        t0 = time.perf_counter()

        # Optional pruning of stale tracks
        if cleanup_idle:
            self.buffer.cleanup_stale_tracks(current_frame=frame_num)

        # 1. Update buffer for all observations in this frame
        buffer_states: List[TrackBufferState] = []
        for obs in track_observations:
            if "frame" not in obs:
                obs["frame"] = frame_num
            buffer_states.append(self.buffer.update(obs))

        results = [None] * len(buffer_states)
        ready_indices = []
        ready_sequences = []

        # 2. Separate ready vs warming-up tracks
        for idx, b_state in enumerate(buffer_states):
            if b_state.is_ready:
                ready_indices.append(idx)
                ready_sequences.append(b_state.sequence)
            else:
                results[idx] = {
                    "track_id": b_state.track_id,
                    "prediction_ready": False,
                    "status": "WARMING_UP",
                    "warmup_progress": b_state.warmup_progress,
                    "observations_recorded": b_state.observation_count,
                    "observations_required": self.sequence_length,
                    "predictions": None,
                    "confidence": 0.0,
                    "is_uncertain": True,
                    "safety_advisory": "Trajectory warming up — collecting temporal history.",
                    "timestamp": b_state.last_time,
                    "zone_id": b_state.zone_id,
                    "lane_id": b_state.lane_id,
                }

        # 3. Batched inference for all ready sequences
        if ready_sequences:
            batch_arr = np.stack(ready_sequences, axis=0)  # (B, 20, 20)
            tensor_batch = self.preprocess_sequence(batch_arr)

            with torch.no_grad():
                raw_outputs = self.model(tensor_batch)

            for b_idx, orig_idx in enumerate(ready_indices):
                b_state = buffer_states[orig_idx]
                preds, conf, is_unc = self._decode_predictions(raw_outputs, idx=b_idx)
                advisory = self._generate_safety_advisory(preds, is_unc)

                results[orig_idx] = {
                    "track_id": b_state.track_id,
                    "prediction_ready": True,
                    "status": "READY",
                    "warmup_progress": 1.0,
                    "observations_recorded": b_state.observation_count,
                    "observations_required": self.sequence_length,
                    "predictions": preds,
                    "confidence": conf,
                    "is_uncertain": is_unc,
                    "safety_advisory": advisory,
                    "timestamp": b_state.last_time,
                    "zone_id": b_state.zone_id,
                    "lane_id": b_state.lane_id,
                }

        total_frame_ms = (time.perf_counter() - t0) * 1000.0
        self._update_telemetry(total_frame_ms)

        return results

    def _generate_safety_advisory(self, preds: Dict[str, Any], is_uncertain: bool) -> str:
        """Constructs human-interpretable safety advisory from predictions."""
        if is_uncertain:
            return "Prediction uncertain — maintain standard traffic surveillance."

        risk = preds.get("risk_level", {}).get("label", "SAFE")
        app = preds.get("is_approaching", {}).get("is_approaching", False)
        inf = preds.get("has_infraction", {}).get("violation_detected", False)
        inf_type = preds.get("infraction_type", {}).get("label", "NONE")

        if risk == "HIGH":
            if inf:
                return f"CRITICAL HAZARD: High risk with active {inf_type} infraction detected!"
            elif app:
                return "WARNING: High-speed vehicle approaching intersection zone rapidly."
            return "HIGH RISK: Aggressive maneuvering or rapid deceleration in corridor."
        elif risk == "WARNING":
            return "MONITOR: Elevated risk profile observed along trajectory."
        return "NOMINAL: Vehicle operating within safe kinematic boundaries."

    def _update_telemetry(self, latency_ms: float) -> None:
        """Records inference runtime statistics."""
        self.telemetry["total_inferences"] += 1
        self.telemetry["total_latency_ms"] += latency_ms
        self.telemetry["last_latency_ms"] = round(latency_ms, 2)
        if latency_ms < self.telemetry["min_latency_ms"]:
            self.telemetry["min_latency_ms"] = round(latency_ms, 2)
        if latency_ms > self.telemetry["max_latency_ms"]:
            self.telemetry["max_latency_ms"] = round(latency_ms, 2)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns runtime performance statistics."""
        total = max(1, self.telemetry["total_inferences"])
        avg_lat = self.telemetry["total_latency_ms"] / total
        return {
            "device": str(self.device),
            "total_inferences": self.telemetry["total_inferences"],
            "average_latency_ms": round(avg_lat, 2),
            "min_latency_ms": self.telemetry["min_latency_ms"],
            "max_latency_ms": self.telemetry["max_latency_ms"],
            "last_latency_ms": self.telemetry["last_latency_ms"],
            "approximate_fps": round(1000.0 / avg_lat, 1) if avg_lat > 0 else 0.0,
            "active_tracks_in_buffer": len(self.buffer),
        }
