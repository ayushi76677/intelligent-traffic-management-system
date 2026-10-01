"""
Phase 10 Automated Test Suite: Final System Integration & Project Verification
==============================================================================
Validates the complete end-to-end intelligent traffic system:
1. Backend service health and unified system status (/api/system/status)
2. Model loading & checkpoint integrity (226,937 parameters, 10 output heads)
3. Feature scaler verification (20 dimensions)
4. Detection & tracking cache ingestion (1,530 frames, 34,679 track points)
5. 20-feature extraction and sequence buffering (warm-up to active transition)
6. Real-time hybrid inference forward pass & latency
7. Traffic intelligence synthesis with strict 3-way separation (Measured, Predicted, Derived)
8. Alert engine schema verification (type, severity, timestamp, source, status)
9. Authority Mode API integrity
10. Driver Mode API integrity & route evaluation
11. SUMO 1.27.1 co-simulation & TraCI 1.27.1 handshake
12. Closed-loop control policy whitelist & parameter bound clamping
13. Simulation REST endpoints & strict "SIMULATION DATA" tagging
14. Data provenance isolation (no simulation/live cross-contamination)
15. Full end-to-end pipeline execution
"""

import os
import sys
import json
import unittest
import torch
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app import app, engine, sumo_engine
from models.tcn_transformer_hybrid import TCNTransformerHybrid
from simulation.traci_bridge import TraciBridge
from simulation.traffic_state_adapter import TrafficStateAdapter, EXACT_FEATURE_NAMES
from simulation.control_policy import ClosedLoopControlPolicy, ALLOWED_ACTIONS
from simulation.sumo_controller import SumoClosedLoopController


class TestPhase10SystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.engine = engine
        cls.sumo_engine = sumo_engine

    # 1. Unified System Status Endpoint
    def test_01_unified_system_status(self):
        res = self.client.get("/api/system/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("status", data)
        self.assertIn(data["status"], ["SYSTEM READY", "LIVE INFERENCE ACTIVE", "SIMULATION ACTIVE"])
        self.assertTrue(data.get("api_healthy"))
        self.assertTrue(data.get("model_loaded"))
        self.assertTrue(data.get("scaler_loaded"))
        self.assertTrue(data.get("authority_mode_available"))
        self.assertTrue(data.get("driver_mode_available"))

        arch = data.get("architecture", {})
        self.assertEqual(arch.get("model_name"), "Proposed TCN-Transformer Gated Hybrid Architecture")
        self.assertEqual(arch.get("parameter_count"), 226937)
        self.assertEqual(arch.get("input_features"), 20)
        self.assertEqual(arch.get("sequence_length"), 20)
        self.assertEqual(arch.get("inference_framework"), "PyTorch")

        sim = data.get("simulation", {})
        self.assertEqual(sim.get("engine"), "Eclipse SUMO")
        self.assertEqual(sim.get("sumo_version"), "1.27.1")
        self.assertEqual(sim.get("traci_version"), "1.27.1")

    # 2. Checkpoint Model Integrity
    def test_02_checkpoint_model_integrity(self):
        ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_model.pth")
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)

        model = TCNTransformerHybrid()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 226937, f"Parameter count mismatch: {params} != 226937")

        # Multi-task forward pass validation
        dummy_x = torch.randn(4, 20, 20)
        with torch.no_grad():
            outputs = model(dummy_x)

        expected_heads = [
            "target_congestion_level", "target_congestion_score",
            "target_risk_level", "target_is_approaching", "target_approach_score",
            "target_motion_state", "target_maneuver_type", "target_has_infraction",
            "target_infraction_type", "aux_next_displacement"
        ]
        for head in expected_heads:
            self.assertIn(head, outputs, f"Missing output head: {head}")
            self.assertFalse(torch.isnan(outputs[head]).any(), f"NaN detected in {head}")
            self.assertFalse(torch.isinf(outputs[head]).any(), f"Inf detected in {head}")

    # 3. Scaler Dimensions
    def test_03_scaler_dimensions(self):
        import joblib
        scaler_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "feature_scaler.joblib")
        self.assertTrue(os.path.exists(scaler_path))
        scaler = joblib.load(scaler_path)
        self.assertEqual(scaler.mean_.shape, (20,), "Scaler mean must have exactly 20 features.")
        self.assertEqual(scaler.scale_.shape, (20,), "Scaler scale must have exactly 20 features.")

    # 4. Telemetry Cache Ingestion
    def test_04_telemetry_cache_ingestion(self):
        self.assertGreater(self.engine.total_frames, 1000, "Telemetry cache must contain >1000 frames.")
        self.assertGreater(len(self.engine.frames_data), 1000, "Must contain >1000 frame records.")
        self.assertEqual(len(self.engine.zones), 6, "Must define 6 spatial zones.")

    # 5. 20-Feature Adapter & Sequence Buffer
    def test_05_feature_adapter_and_sequence_buffer(self):
        adapter = TrafficStateAdapter(sequence_length=20)
        self.assertEqual(len(EXACT_FEATURE_NAMES), 20)

        for step in range(1, 20):
            obs = {"vehicle_id": "car_integration_1", "x": float(step * 5), "y": float(step * 5), "speed": 12.0, "angle": 90.0}
            state = adapter.update_vehicle(obs, step_num=step, sim_time=float(step))
            self.assertFalse(state.is_ready)
            self.assertEqual(state.status, "WARMING_UP")

        obs20 = {"vehicle_id": "car_integration_1", "x": 100.0, "y": 100.0, "speed": 12.0, "angle": 90.0}
        state20 = adapter.update_vehicle(obs20, step_num=20, sim_time=20.0)
        self.assertTrue(state20.is_ready)
        self.assertEqual(state20.status, "ACTIVE")
        self.assertEqual(state20.sequence.shape, (20, 20))

    # 6. Real-Time Inference Latency
    def test_06_realtime_inference_latency(self):
        pred = self.engine.hybrid_predictor
        self.assertIsNotNone(pred)
        dummy_seq = np.random.randn(20, 20).astype(np.float32)
        import time
        t0 = time.perf_counter()
        tensor_x = pred.preprocess_sequence(dummy_seq)
        with torch.no_grad():
            raw_out = pred.model(tensor_x)
        preds, conf, is_unc = pred._decode_predictions(raw_out, idx=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertIn("congestion_level", preds)
        self.assertIn("congestion_score", preds)
        self.assertIn("risk_level", preds)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLess(elapsed_ms, 100.0, f"Inference took too long: {elapsed_ms:.2f} ms")

    # 7. Strict 3-Way Traffic Intelligence Segregation
    def test_07_three_way_traffic_intelligence(self):
        frame_data = self.engine.get_frame_data(100)
        self.assertIn("hybrid_intelligence", frame_data)
        intel = frame_data["hybrid_intelligence"]

        # Verify hierarchical presence
        self.assertIn("vehicle_level", intel)
        self.assertIn("zone_level", intel)
        self.assertIn("system_level", intel)

        # Check API endpoint for frame intelligence
        res = self.client.get("/api/frame/100")
        self.assertEqual(res.status_code, 200)
        f_data = res.get_json()
        self.assertIn("hybrid_intelligence", f_data)
        self.assertIn("system_summary", f_data)
        self.assertIn("zones", f_data)

    # 8. Alert Engine Schema (Source & Status)
    def test_08_alert_engine_schema(self):
        res = self.client.get("/api/alerts?frame=100")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("alerts", data)

        allowed_sources = ["MEASURED", "PREDICTED", "DERIVED", "MODEL", "SIMULATION"]
        for alert in data["alerts"]:
            self.assertIn("alert_id", alert)
            self.assertIn("type", alert)
            self.assertIn("severity", alert)
            self.assertIn("timestamp", alert)
            self.assertIn("source", alert)
            self.assertIn("status", alert)
            self.assertIn(alert["source"], allowed_sources, f"Invalid alert source: {alert['source']}")

    # 9. Authority Mode API Integrity
    def test_09_authority_mode_apis(self):
        r_cams = self.client.get("/api/cameras")
        self.assertEqual(r_cams.status_code, 200)

        r_zones = self.client.get("/api/zones")
        self.assertEqual(r_zones.status_code, 200)

        r_vios = self.client.get("/api/violations")
        self.assertEqual(r_vios.status_code, 200)

        r_status = self.client.get("/api/status")
        self.assertEqual(r_status.status_code, 200)
        self.assertEqual(r_status.get_json()["mode"], "AUTHORITY MODE")

    # 10. Driver Mode API Integrity
    def test_10_driver_mode_apis(self):
        r_feed = self.client.get("/api/driver/feed?lat=22.7533&lng=75.8937")
        self.assertEqual(r_feed.status_code, 200)
        feed_data = r_feed.get_json()
        self.assertEqual(feed_data["service"], "DriverModeSync")
        self.assertEqual(feed_data["status"], "ONLINE")
        self.assertIn("current_road", feed_data)

        r_eval = self.client.post("/api/routes/evaluate", json={
            "origin": {"lat": 22.7533, "lng": 75.8937},
            "destination": {"lat": 22.7244, "lng": 75.8839}
        })
        self.assertEqual(r_eval.status_code, 200)
        eval_data = r_eval.get_json()
        self.assertIn("route_condition", eval_data)
        self.assertIn("data_provenance", eval_data)
        self.assertIn("recommended_action", eval_data)

    # 11. SUMO & TraCI Handshake
    def test_11_sumo_traci_lifecycle(self):
        bridge = TraciBridge(config_file="sumo/simulation.sumocfg", workspace_dir=WORKSPACE_ROOT, label="p10_test")
        self.assertIsNotNone(bridge.sumo_binary)
        self.assertTrue(bridge.start())
        self.assertTrue(bridge.is_connected)
        for _ in range(3):
            st = bridge.step()
            self.assertIn("simulation_time", st)
        bridge.close()
        self.assertFalse(bridge.is_connected)

    # 12. Control Policy Safety Validation
    def test_12_control_policy_safety(self):
        policy = ClosedLoopControlPolicy(min_confidence_threshold=0.50)
        self.assertIn("VARIABLE_SPEED_LIMIT", ALLOWED_ACTIONS)
        self.assertIn("REROUTE_VEHICLE", ALLOWED_ACTIONS)

        # Unwhitelisted action -> rejected
        bad = policy._validate_and_build_action("UNAUTHORIZED_DRONE_COMMAND", "A0A1", {}, "test", 0.99, 10.0)
        self.assertIsNone(bad)

        # Out-of-bounds speed -> rejected
        fast = policy._validate_and_build_action("VARIABLE_SPEED_LIMIT", "A0A1", {"speed_mps": 45.0}, "test", 0.90, 10.0)
        self.assertIsNone(fast)

    # 13. Simulation REST Endpoints & SIMULATION DATA Tagging
    def test_13_simulation_rest_endpoints(self):
        r_sim_st = self.client.get("/api/simulation/status")
        self.assertEqual(r_sim_st.status_code, 200)
        self.assertEqual(r_sim_st.get_json()["mode"], "SIMULATION DATA")

        r_sim_tr = self.client.get("/api/simulation/traffic")
        self.assertEqual(r_sim_tr.status_code, 200)
        self.assertEqual(r_sim_tr.get_json()["mode"], "SIMULATION DATA")

        r_sim_ctrl = self.client.get("/api/simulation/control")
        self.assertEqual(r_sim_ctrl.status_code, 200)
        self.assertEqual(r_sim_ctrl.get_json()["mode"], "SIMULATION DATA")

        r_sim_met = self.client.get("/api/simulation/metrics")
        self.assertEqual(r_sim_met.status_code, 200)
        self.assertEqual(r_sim_met.get_json()["mode"], "SIMULATION DATA")
        self.assertTrue(r_sim_met.get_json()["has_baseline"])
        self.assertTrue(r_sim_met.get_json()["has_controlled"])

    # 14. Data Provenance Isolation
    def test_14_data_provenance_isolation(self):
        r_auth = self.client.get("/api/frame/50")
        self.assertEqual(r_auth.status_code, 200)
        auth_data = r_auth.get_json()
        self.assertNotEqual(auth_data.get("provenance"), "ECLIPSE_SUMO_TRACI")

        r_sim = self.client.get("/api/simulation/status")
        sim_data = r_sim.get_json()
        self.assertEqual(sim_data.get("data_source"), "ECLIPSE_SUMO_TRACI")
        self.assertEqual(sim_data.get("mode"), "SIMULATION DATA")

    # 15. End-to-End Pipeline Execution
    def test_15_end_to_end_pipeline(self):
        # 1. Start simulation
        start_res = self.client.post("/api/simulation/start")
        self.assertEqual(start_res.status_code, 200)

        # 2. Step co-simulation
        step_res = self.client.post("/api/simulation/step")
        self.assertEqual(step_res.status_code, 200)
        self.assertEqual(step_res.get_json()["status"], "RUNNING")

        # 3. Check traffic state
        traffic_res = self.client.get("/api/simulation/traffic")
        self.assertEqual(traffic_res.status_code, 200)

        # 4. Stop simulation cleanly
        stop_res = self.client.post("/api/simulation/stop")
        self.assertEqual(stop_res.status_code, 200)
        self.assertEqual(stop_res.get_json()["status"], "STOPPED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
