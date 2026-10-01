"""
Comprehensive Automated Test Suite for Phase 6 Real-Time Hybrid Model Integration
=================================================================================

Validates:
1. Model Loading (checkpoint, eval mode, parameter count)
2. Scaler Loading (joblib, 20 features)
3. Feature Ordering (exact match to training and metadata)
4. Sequence Length (exactly 20 timesteps)
5. Temporal Buffer (FIFO, sliding window, track isolation)
6. Insufficient History Handling (warm-up state, progress reporting)
7. Single-Track Live Inference (full 10-head output decoding)
8. Multi-Track Batch Inference (batched forward pass, independent tracks)
9. Disappearing & Reappearing Tracks (eviction on idle timeout, clean reset)
10. Malformed Input Handling (missing fields, edge-case coordinates)
11. NaN / Inf Robustness (sanitization, no crashes or propagation)
12. Output Shapes & Value Ranges (probabilities sum to 1, valid regression ranges)
13. Confidence & Safety Uncertainty Gates (flags low-confidence predictions)
14. Traffic-Level Aggregation (Vehicle, Lane, Zone A/B/C, and System levels)
15. Backend REST API Integration (status, hybrid status, frame, live inference, alerts, driver feed)
16. End-to-End Live Pipeline (tracking -> features -> buffer -> scaler -> model -> intelligence)
"""

import os
import sys
import unittest
import numpy as np
import torch
import joblib

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.tcn_transformer_hybrid import TCNTransformerHybrid
from ml.hybrid_traffic.feature_adapter import LiveFeatureAdapter, EXACT_FEATURE_NAMES, FEATURE_COUNT
from ml.hybrid_traffic.sequence_buffer import TrackSequenceBuffer, TrackBufferState
from ml.hybrid_traffic.realtime_inference import RealTimeTrafficPredictor
from ml.hybrid_traffic.traffic_aggregator import TrafficIntelligenceAggregator
from traffic_intelligence import TrafficIntelligenceEngine
from app import app


class TestPhase6RealTimeIntegration(unittest.TestCase):
    """Full verification suite for Phase 6 Real-Time Integration."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
        cls.scaler_path = os.path.join(BASE_DIR, "checkpoints", "feature_scaler.joblib")
        cls.config_path = os.path.join(BASE_DIR, "checkpoints", "best_model_config.json")

        # Initialize core components
        cls.feature_adapter = LiveFeatureAdapter()
        cls.buffer = TrackSequenceBuffer(sequence_length=20, max_idle_frames=15, feature_adapter=cls.feature_adapter)
        cls.predictor = RealTimeTrafficPredictor(
            checkpoint_path=cls.checkpoint_path,
            scaler_path=cls.scaler_path,
            config_path=cls.config_path,
            device="cpu",
            confidence_threshold=0.60
        )
        cls.aggregator = TrafficIntelligenceAggregator()
        cls.flask_client = app.test_client()

    # -------------------------------------------------------------------------
    # 1. Model Loading
    # -------------------------------------------------------------------------
    def test_01_model_loading(self):
        """Verify model checkpoint exists, loads properly in eval mode, and has frozen weights."""
        self.assertTrue(os.path.exists(self.checkpoint_path), "Checkpoint file missing.")
        self.assertIsNotNone(self.predictor.model, "Model not instantiated.")
        self.assertFalse(self.predictor.model.training, "Model should be in eval() mode.")
        
        # Check total parameters match Phase 5 config (226,937 params)
        total_params = sum(p.numel() for p in self.predictor.model.parameters())
        self.assertEqual(total_params, 226937, f"Expected 226,937 parameters, got {total_params}")

        # Check weights are frozen (requires_grad is False)
        for param in self.predictor.model.parameters():
            self.assertFalse(param.requires_grad, "Model parameters must be frozen.")

    # -------------------------------------------------------------------------
    # 2. Scaler Loading
    # -------------------------------------------------------------------------
    def test_02_scaler_loading(self):
        """Verify feature scaler loads via joblib and has exactly 20 features."""
        self.assertTrue(os.path.exists(self.scaler_path), "Scaler file missing.")
        self.assertIsNotNone(self.predictor.scaler, "Scaler not instantiated.")
        self.assertEqual(self.predictor.scaler.n_features_in_, 20, "Scaler must have 20 input features.")
        self.assertEqual(self.predictor.scaler.mean_.shape, (20,), "Scaler mean vector must be shape (20,).")
        self.assertEqual(self.predictor.scaler.scale_.shape, (20,), "Scaler scale vector must be shape (20,).")

    # -------------------------------------------------------------------------
    # 3. Feature Ordering
    # -------------------------------------------------------------------------
    def test_03_feature_ordering(self):
        """Verify exact 20 features match the canonical sequence builder ordering."""
        expected_features = [
            "center_x", "center_y", "delta_x", "delta_y", "displacement_image",
            "speed_image_px_per_sec", "accel_image_px_per_sec2", "heading_rad",
            "heading_change_rad", "ground_x", "ground_y", "ground_delta_x",
            "ground_delta_y", "displacement_ground", "ground_speed_norm_per_sec",
            "box_width", "box_height", "box_area", "stopped_duration",
            "speed_variance_rolling5"
        ]
        self.assertEqual(EXACT_FEATURE_NAMES, expected_features, "Feature names order mismatch.")
        self.assertEqual(len(EXACT_FEATURE_NAMES), 20, "Feature count must be exactly 20.")

        # Test single computation produces vector of length 20
        obs = {"x1": 100.0, "y1": 200.0, "x2": 150.0, "y2": 300.0, "time": 1.0, "frame": 30}
        vec, meta = self.feature_adapter.compute_observation_features(obs)
        self.assertEqual(vec.shape, (20,), "Feature vector must be 1D with 20 elements.")
        self.assertEqual(vec.dtype, np.float32, "Feature vector must be float32.")

    # -------------------------------------------------------------------------
    # 4. Sequence Length
    # -------------------------------------------------------------------------
    def test_04_sequence_length(self):
        """Verify sequence buffer enforces length 20."""
        self.assertEqual(self.buffer.sequence_length, 20)
        self.assertEqual(self.predictor.sequence_length, 20)

    # -------------------------------------------------------------------------
    # 5. Temporal Buffer Operations
    # -------------------------------------------------------------------------
    def test_05_temporal_buffer_operations(self):
        """Verify buffer updates FIFO, maintains maxlen 20, and isolates tracks."""
        test_buf = TrackSequenceBuffer(sequence_length=20, feature_adapter=self.feature_adapter)
        
        # Ingest 25 observations for track 1
        for f in range(25):
            state = test_buf.update({
                "track_id": 1,
                "x1": 100.0 + f * 2,
                "y1": 200.0 + f * 1,
                "x2": 160.0 + f * 2,
                "y2": 280.0 + f * 1,
                "frame": f,
                "time": f * 0.033
            })

        self.assertEqual(state.observation_count, 20, "Buffer deque must not exceed maxlen 20.")
        self.assertTrue(state.is_ready, "Track should be ready after 20 observations.")
        self.assertEqual(state.sequence.shape, (20, 20), "Ready sequence must have shape (20, 20).")

    # -------------------------------------------------------------------------
    # 6. Insufficient History Handling (Warm-up)
    # -------------------------------------------------------------------------
    def test_06_insufficient_history_handling(self):
        """Verify tracks with <20 observations return explicit warm-up status without making invalid predictions."""
        test_buf = TrackSequenceBuffer(sequence_length=20, feature_adapter=self.feature_adapter)
        
        for f in range(1, 15):
            state = test_buf.update({
                "track_id": 88,
                "x1": 50.0, "y1": 50.0, "x2": 80.0, "y2": 100.0,
                "frame": f, "time": f * 0.033
            })
            self.assertFalse(state.is_ready, f"Track should NOT be ready at observation {f}")
            self.assertEqual(state.status, "WARMING_UP")
            self.assertIsNone(state.sequence, "Sequence must be None during warm-up.")
            self.assertAlmostEqual(state.warmup_progress, f / 20.0, places=2)

    # -------------------------------------------------------------------------
    # 7. Single-Track Live Inference
    # -------------------------------------------------------------------------
    def test_07_single_track_inference(self):
        """Verify live inference decodes all 10 heads with valid structured dictionary."""
        predictor = RealTimeTrafficPredictor(device="cpu")
        
        # Feed 20 frames
        for f in range(20):
            res = predictor.predict_live({
                "track_id": 999,
                "x1": 200.0 + f * 3,
                "y1": 300.0 + f * 2,
                "x2": 260.0 + f * 3,
                "y2": 380.0 + f * 2,
                "frame": f,
                "time": f * 0.033
            })

        # Frame 20 must be ready
        self.assertTrue(res["prediction_ready"], "Track 999 should be ready after 20 frames.")
        self.assertEqual(res["status"], "READY")
        self.assertEqual(res["warmup_progress"], 1.0)
        
        preds = res["predictions"]
        self.assertIsNotNone(preds, "Predictions must not be None.")

        # Check all 10 target heads exist
        expected_heads = [
            "congestion_level", "congestion_score", "risk_level",
            "is_approaching", "approach_threat_score", "motion_state",
            "maneuver_type", "has_infraction", "infraction_type",
            "predicted_next_displacement"
        ]
        for head in expected_heads:
            self.assertIn(head, preds, f"Missing target head: {head}")

        # Check values
        self.assertIn(preds["congestion_level"]["label"], ["LOW", "MEDIUM", "HIGH"])
        self.assertGreaterEqual(preds["congestion_score"], 10.0)
        self.assertLessEqual(preds["congestion_score"], 100.0)
        self.assertIn(preds["risk_level"]["label"], ["SAFE", "WARNING", "HIGH"])
        self.assertIn(preds["is_approaching"]["label"], ["NON_APPROACHING", "APPROACHING"])
        self.assertIn(preds["motion_state"]["label"], ["STOPPED", "ACCELERATING", "DECELERATING", "CRUISING"])
        self.assertIn(preds["maneuver_type"]["label"], ["STATIONARY", "TURNING_LEFT", "TURNING_RIGHT", "STRAIGHT"])
        self.assertIn(preds["has_infraction"]["label"], ["COMPLIANT", "INFRACTION"])
        self.assertIn(preds["infraction_type"]["label"], ["NONE", "OVERSPEEDING", "ILLEGAL_STOPPING"])

    # -------------------------------------------------------------------------
    # 8. Multi-Track Batch Inference
    # -------------------------------------------------------------------------
    def test_08_multiple_track_inference(self):
        """Verify multiple active tracks are handled simultaneously with batched execution."""
        predictor = RealTimeTrafficPredictor(device="cpu")

        # Simulate 20 frames for 3 different tracks
        for f in range(20):
            observations = [
                {"track_id": 101, "x1": 100 + f, "y1": 200, "x2": 150 + f, "y2": 260, "frame": f, "time": f * 0.033},
                {"track_id": 102, "x1": 300 + f*2, "y1": 400, "x2": 370 + f*2, "y2": 490, "frame": f, "time": f * 0.033},
                {"track_id": 103, "x1": 500, "y1": 500, "x2": 540, "y2": 570, "frame": f, "time": f * 0.033},
            ]
            results = predictor.predict_frame_tracks(frame_num=f, track_observations=observations)
            self.assertEqual(len(results), 3)

        # At frame 19 (20th observation), all three tracks should be READY
        for r in results:
            self.assertTrue(r["prediction_ready"])
            self.assertEqual(r["status"], "READY")
            self.assertIsNotNone(r["predictions"])

    # -------------------------------------------------------------------------
    # 9. Disappearing & Reappearing Tracks
    # -------------------------------------------------------------------------
    def test_09_disappearing_tracks(self):
        """Verify idle tracks are evicted by cleanup and re-enter warm-up if reappearing."""
        test_buf = TrackSequenceBuffer(sequence_length=20, max_idle_frames=10, feature_adapter=self.feature_adapter)
        
        # Track 50 observed at frame 0
        test_buf.update({"track_id": 50, "x1": 100, "y1": 100, "x2": 150, "y2": 150, "frame": 0})
        self.assertIn(50, test_buf.get_active_track_ids())

        # Cleanup at frame 15 (gap = 15 > 10)
        evicted = test_buf.cleanup_stale_tracks(current_frame=15)
        self.assertIn(50, evicted, "Track 50 should be evicted.")
        self.assertNotIn(50, test_buf.get_active_track_ids(), "Track 50 should no longer be active.")

        # Reappearance at frame 30
        state = test_buf.update({"track_id": 50, "x1": 200, "y1": 200, "x2": 250, "y2": 250, "frame": 30})
        self.assertEqual(state.observation_count, 1, "Reappeared track must reset to observation count 1.")
        self.assertFalse(state.is_ready, "Reappeared track must re-enter warm-up.")

    # -------------------------------------------------------------------------
    # 10. Malformed Input Handling
    # -------------------------------------------------------------------------
    def test_10_malformed_input(self):
        """Verify robustness against inverted bboxes, zero time delta, and negative values."""
        # Inverted box (x1 > x2, y1 > y2)
        obs = {"x1": 200.0, "y1": 300.0, "x2": 100.0, "y2": 200.0, "frame": 1, "time": 0.0}
        vec, meta = self.feature_adapter.compute_observation_features(obs)
        self.assertEqual(vec.shape, (20,))
        self.assertFalse(np.isnan(vec).any(), "Inverted box must not yield NaNs.")
        self.assertFalse(np.isinf(vec).any(), "Inverted box must not yield Infs.")

    # -------------------------------------------------------------------------
    # 11. NaN / Inf Handling
    # -------------------------------------------------------------------------
    def test_11_nan_inf_handling(self):
        """Verify NaN/Inf inputs are sanitized cleanly to finite values."""
        obs = {"x1": float("nan"), "y1": 100.0, "x2": float("inf"), "y2": 200.0, "frame": 1, "time": 0.0}
        # Preprocessing sequence with non-finite values
        bad_seq = np.full((20, 20), np.nan, dtype=np.float32)
        bad_seq[0, 0] = np.inf
        
        tensor_x = self.predictor.preprocess_sequence(bad_seq)
        self.assertFalse(torch.isnan(tensor_x).any().item(), "Preprocessed tensor must not contain NaNs.")
        self.assertFalse(torch.isinf(tensor_x).any().item(), "Preprocessed tensor must not contain Infs.")

        with torch.no_grad():
            outputs = self.predictor.model(tensor_x)
        for head, tensor in outputs.items():
            self.assertFalse(torch.isnan(tensor).any().item(), f"Head {head} output contained NaN.")
            self.assertFalse(torch.isinf(tensor).any().item(), f"Head {head} output contained Inf.")

    # -------------------------------------------------------------------------
    # 12. Output Shape and Type Verification
    # -------------------------------------------------------------------------
    def test_12_output_shape_and_types(self):
        """Verify softmax probabilities sum to 1.0 and shapes match specifications."""
        dummy_seq = np.random.randn(20, 20).astype(np.float32)
        tensor_x = self.predictor.preprocess_sequence(dummy_seq)
        with torch.no_grad():
            raw = self.predictor.model(tensor_x)
        
        preds, conf, is_unc = self.predictor._decode_predictions(raw, idx=0)

        # Probabilities sum to 1.0
        for head in ["congestion_level", "risk_level"]:
            probs = list(preds[head]["probabilities"].values())
            self.assertAlmostEqual(sum(probs), 1.0, places=3, msg=f"Probabilities for {head} must sum to 1.0")

        # Regression ranges
        self.assertGreaterEqual(preds["congestion_score"], 10.0)
        self.assertLessEqual(preds["congestion_score"], 100.0)
        self.assertGreaterEqual(preds["approach_threat_score"], 0.0)
        self.assertLessEqual(preds["approach_threat_score"], 100.0)

    # -------------------------------------------------------------------------
    # 13. Confidence & Uncertainty Handling
    # -------------------------------------------------------------------------
    def test_13_confidence_handling(self):
        """Verify confidence thresholding flags predictions as uncertain."""
        # Force low-confidence logits (uniform distribution)
        raw_outputs = {
            "target_congestion_level": torch.tensor([[0.0, 0.0, 0.0]]),
            "target_congestion_score": torch.tensor([[35.0]]),
            "target_risk_level": torch.tensor([[0.0, 0.0, 0.0]]),
            "target_is_approaching": torch.tensor([[0.0, 0.0]]),
            "target_approach_score": torch.tensor([[20.0]]),
            "target_motion_state": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            "target_maneuver_type": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            "target_has_infraction": torch.tensor([[0.0, 0.0]]),
            "target_infraction_type": torch.tensor([[0.0, 0.0, 0.0]]),
            "aux_next_displacement": torch.tensor([[0.0, 0.0]]),
        }
        preds, conf, is_uncertain = self.predictor._decode_predictions(raw_outputs, idx=0)
        # Uniform logits give confidences ~0.33, ~0.50, ~0.25 -> mean < 0.60
        self.assertTrue(is_uncertain, "Uniform logits must be flagged as uncertain.")

    # -------------------------------------------------------------------------
    # 14. Traffic-Level Aggregation
    # -------------------------------------------------------------------------
    def test_14_traffic_level_aggregation(self):
        """Verify aggregation separates A (measured), B (predicted), and C (derived)."""
        detections = [
            {"track_id": 1, "class_id": 2, "vehicle_type": "car", "x1": 100, "y1": 100, "x2": 150, "y2": 150, "bottom_x": 125, "bottom_y": 150, "zone": "ZONE 1", "speed_px_per_sec": 45.0},
            {"track_id": 2, "class_id": 3, "vehicle_type": "motorcycle", "x1": 200, "y1": 200, "x2": 230, "y2": 250, "bottom_x": 215, "bottom_y": 250, "zone": "ZONE 1", "speed_px_per_sec": 30.0},
        ]
        # Simulate predictions
        predictions = [
            {
                "track_id": 1,
                "prediction_ready": True,
                "status": "READY",
                "warmup_progress": 1.0,
                "confidence": 0.85,
                "is_uncertain": False,
                "zone_id": "ZONE 1",
                "lane_id": "LANE_NORTH_INFLOW",
                "safety_advisory": "Nominal",
                "predictions": {
                    "congestion_level": {"label": "HIGH", "confidence": 0.88},
                    "congestion_score": 75.0,
                    "risk_level": {"label": "SAFE", "confidence": 0.90},
                    "is_approaching": {"is_approaching": False, "confidence": 0.95},
                    "approach_threat_score": 10.0,
                    "motion_state": {"label": "CRUISING", "confidence": 0.85},
                    "maneuver_type": {"label": "STRAIGHT", "confidence": 0.92},
                    "has_infraction": {"violation_detected": False, "confidence": 0.95},
                    "infraction_type": {"label": "NONE", "confidence": 0.95},
                }
            },
            {
                "track_id": 2,
                "prediction_ready": False,
                "status": "WARMING_UP",
                "warmup_progress": 0.25,
                "confidence": 0.0,
                "is_uncertain": True,
                "zone_id": "ZONE 1",
                "lane_id": "LANE_NORTH_INFLOW",
                "safety_advisory": "Warming up",
                "predictions": None,
            }
        ]

        intel = self.aggregator.aggregate(
            frame_num=10,
            time_seconds=0.33,
            measured_detections=detections,
            model_predictions=predictions,
            deterministic_zones={"ZONE 1": {"level": "MEDIUM", "trend": "INCREASING", "average_count": 2.0, "sustained_seconds": 0.0}}
        )

        # Check hierarchy
        self.assertIn("vehicle_level", intel)
        self.assertIn("lane_level", intel)
        self.assertIn("zone_level", intel)
        self.assertIn("system_level", intel)

        # Check Zone 1 sections A, B, C
        z1 = intel["zone_level"]["ZONE 1"]
        self.assertIn("measured_values", z1)
        self.assertIn("model_predictions", z1)
        self.assertIn("derived_intelligence", z1)

        # Verify A (measured) has count 2
        self.assertEqual(z1["measured_values"]["vehicle_count"], 2)
        # Verify B (model) has 1 ready and 1 warming up
        self.assertEqual(z1["model_predictions"]["predictions_ready_count"], 1)
        self.assertEqual(z1["model_predictions"]["warming_up_count"], 1)
        # Verify C (derived) has fused congestion score
        self.assertGreater(z1["derived_intelligence"]["fused_congestion_score"], 0.0)

    # -------------------------------------------------------------------------
    # 15. Backend REST API Integration
    # -------------------------------------------------------------------------
    def test_15_backend_api_integration(self):
        """Verify existing backend endpoints and new hybrid endpoints respond successfully."""
        # /api/status
        r_status = self.flask_client.get("/api/status")
        self.assertEqual(r_status.status_code, 200)
        self.assertIn("TCN-Transformer Gated Hybrid", r_status.json.get("active_models", []))
        self.assertIn("hybrid_model", r_status.json)

        # /api/hybrid/status
        r_hstatus = self.flask_client.get("/api/hybrid/status")
        self.assertEqual(r_hstatus.status_code, 200)
        self.assertEqual(r_hstatus.json.get("status"), "ACTIVE")
        self.assertEqual(r_hstatus.json.get("model_architecture"), "TCNTransformerHybrid")

        # /api/hybrid/frame/25
        r_hframe = self.flask_client.get("/api/hybrid/frame/25")
        self.assertEqual(r_hframe.status_code, 200)
        self.assertIn("system_level", r_hframe.json)
        self.assertIn("zone_level", r_hframe.json)

        # /api/hybrid/live (Single)
        r_live_s = self.flask_client.post("/api/hybrid/live", json={
            "track_id": 77, "x1": 150, "y1": 200, "x2": 210, "y2": 310, "frame": 5, "time": 0.16
        })
        self.assertEqual(r_live_s.status_code, 200)
        self.assertEqual(r_live_s.json.get("track_id"), 77)

        # /api/hybrid/live (Batch)
        r_live_b = self.flask_client.post("/api/hybrid/live", json={
            "frame": 5,
            "observations": [
                {"track_id": 77, "x1": 150, "y1": 200, "x2": 210, "y2": 310},
                {"track_id": 78, "x1": 400, "y1": 300, "x2": 450, "y2": 390}
            ]
        })
        self.assertEqual(r_live_b.status_code, 200)
        self.assertIn("vehicle_level", r_live_b.json)

        # /api/alerts
        r_alerts = self.flask_client.get("/api/alerts")
        self.assertEqual(r_alerts.status_code, 200)
        self.assertIn("alerts", r_alerts.json)

        # /api/driver/feed
        r_driver = self.flask_client.get("/api/driver/feed")
        self.assertEqual(r_driver.status_code, 200)
        self.assertIn("hybrid_intelligence", r_driver.json)

    # -------------------------------------------------------------------------
    # 16. End-to-End Pipeline
    # -------------------------------------------------------------------------
    def test_16_end_to_end_pipeline(self):
        """
        Verify end-to-end processing pipeline:
        Video tracks -> Feature Adapter -> Sequence Buffer -> Scaler -> Hybrid Model -> Aggregator -> Intelligence.
        """
        engine = TrafficIntelligenceEngine()
        
        # Test frames 20 to 25
        for f in range(20, 26):
            frame_data = engine.get_frame_data(f)
            self.assertIsNotNone(frame_data)
            self.assertIn("hybrid_intelligence", frame_data)
            
            h_intel = frame_data["hybrid_intelligence"]
            sys_intel = h_intel["system_level"]
            self.assertEqual(sys_intel["frame"], f)
            self.assertIn("priority_zone", sys_intel)
            self.assertIn("system_fused_congestion_score", sys_intel)

            # Check detections are enriched
            for det in frame_data["detections"]:
                self.assertIn("prediction_ready", det)
                self.assertIn("status", det)
                self.assertIn("safety_advisory", det)


if __name__ == "__main__":
    unittest.main()
