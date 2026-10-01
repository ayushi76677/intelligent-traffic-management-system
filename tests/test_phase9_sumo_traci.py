"""
Phase 9 Automated Test Suite: SUMO/TraCI Closed-Loop Traffic Intelligence
========================================================================
Comprehensive verification for all 16 Phase 9 criteria:
1. SUMO startup
2. TraCI connection
3. Vehicle-state extraction
4. Feature mapping (exact 20 features)
5. Sequence buffering (warm-up & ready transitions)
6. Model inference (TCN-Transformer hybrid multi-task forward pass)
7. Traffic intelligence (Measured vs Predicted vs Derived)
8. Control policy (Validation & allowed action whitelist)
9. TraCI control (Validated action dispatch)
10. Simulation continuation (multi-step closed-loop stepping)
11. Baseline simulation results
12. Controlled simulation results
13. Metrics collection & objective comparison
14. Clean shutdown
15. TraCI failure handling
16. Model failure handling & fallback
+ Authority Mode & Driver Mode regression verification
"""

import os
import sys
import json
import unittest
import numpy as np

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simulation.traci_bridge import TraciBridge
from simulation.traffic_state_adapter import TrafficStateAdapter, EXACT_FEATURE_NAMES, FEATURE_COUNT
from simulation.control_policy import ClosedLoopControlPolicy, ValidatedControlAction, ALLOWED_ACTIONS
from simulation.sumo_controller import SumoClosedLoopController
from app import app


class TestPhase9SumoTraciClosedLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace_dir = BASE_DIR
        cls.config_file = "sumo/simulation.sumocfg"
        cls.app = app.test_client()

    # 1. SUMO Startup
    def test_01_sumo_startup(self):
        bridge = TraciBridge(config_file=self.config_file, workspace_dir=self.workspace_dir)
        self.assertIsNotNone(bridge.sumo_binary, "SUMO binary was not located on system.")
        self.assertTrue(os.path.exists(bridge.sumo_binary), f"SUMO binary does not exist at {bridge.sumo_binary}")
        self.assertIn(bridge.status, ["INITIALIZED", "DISCONNECTED"], f"Unexpected bridge status: {bridge.status}")

    # 2. TraCI Connection
    def test_02_traci_connection(self):
        bridge = TraciBridge(config_file=self.config_file, workspace_dir=self.workspace_dir, label="test_conn")
        connected = bridge.start()
        self.assertTrue(connected, "TraCI connection failed to start.")
        self.assertTrue(bridge.is_connected, "TraCI is_connected flag is False.")
        self.assertEqual(bridge.status, "RUNNING", "Bridge status is not RUNNING.")
        self.assertGreaterEqual(bridge.current_sim_time, 0.0, "Simulation time should be >= 0.0.")
        bridge.close()
        self.assertFalse(bridge.is_connected, "TraCI should be disconnected after close().")

    # 3. Vehicle-State Extraction
    def test_03_vehicle_state_extraction(self):
        bridge = TraciBridge(config_file=self.config_file, workspace_dir=self.workspace_dir, label="test_extract")
        self.assertTrue(bridge.start())

        # Advance until vehicles depart
        state = {}
        for _ in range(5):
            state = bridge.step()

        self.assertIn("vehicles", state, "Simulation state missing 'vehicles' dictionary.")
        self.assertIn("edges", state, "Simulation state missing 'edges' dictionary.")
        self.assertIn("simulation_time", state, "Simulation state missing 'simulation_time'.")

        if state["vehicles"]:
            vid, vdata = next(iter(state["vehicles"].items()))
            self.assertIn("x", vdata)
            self.assertIn("y", vdata)
            self.assertIn("speed", vdata)
            self.assertIn("angle", vdata)
            self.assertIn("lane_id", vdata)
            self.assertIn("edge_id", vdata)
            self.assertIn("waiting_time", vdata)
            self.assertIn("vehicle_class", vdata)
        bridge.close()

    # 4. Feature Mapping (Exact 20 Features)
    def test_04_feature_mapping(self):
        adapter = TrafficStateAdapter()
        self.assertEqual(len(EXACT_FEATURE_NAMES), 20, "Exact feature count must be exactly 20.")
        self.assertEqual(FEATURE_COUNT, 20, "FEATURE_COUNT must be 20.")

        dummy_obs = {
            "vehicle_id": "test_veh_1",
            "x": 100.0,
            "y": 100.0,
            "speed": 12.5,
            "angle": 90.0,
            "width": 2.0,
            "length": 5.0,
            "waiting_time": 2.0
        }
        from collections import deque
        recent_speeds = deque(maxlen=5)

        feat_vec, feat_dict = adapter.extract_features(
            obs=dummy_obs,
            prev_obs=None,
            prev_feats=None,
            recent_speeds=recent_speeds,
            dt=1.0
        )

        self.assertEqual(feat_vec.shape, (20,), "Feature vector must have shape (20,).")
        self.assertEqual(len(feat_dict), 20, "Feature dictionary must contain 20 items.")
        self.assertFalse(np.isnan(feat_vec).any(), "Feature vector contains NaN values.")
        self.assertFalse(np.isinf(feat_vec).any(), "Feature vector contains Inf values.")

        for fname in EXACT_FEATURE_NAMES:
            self.assertIn(fname, feat_dict, f"Missing feature: {fname}")

    # 5. Sequence Buffering (Warm-up & Ready Transitions)
    def test_05_sequence_buffering(self):
        adapter = TrafficStateAdapter(sequence_length=20)
        vid = "seq_test_car"

        # Observations 1 to 19: must be WARMING_UP
        for step in range(1, 20):
            obs = {"vehicle_id": vid, "x": float(step * 5), "y": float(step * 5), "speed": 10.0, "angle": 45.0}
            state = adapter.update_vehicle(obs, step_num=step, sim_time=float(step))
            self.assertFalse(state.is_ready, f"Step {step} should not be ready.")
            self.assertEqual(state.status, "WARMING_UP", f"Step {step} status must be WARMING_UP.")
            self.assertIsNone(state.sequence, f"Step {step} sequence must be None.")
            self.assertAlmostEqual(state.warmup_progress, step / 20.0, places=2)

        # Observation 20: must transition to ACTIVE / READY
        obs20 = {"vehicle_id": vid, "x": 100.0, "y": 100.0, "speed": 10.0, "angle": 45.0}
        state20 = adapter.update_vehicle(obs20, step_num=20, sim_time=20.0)
        self.assertTrue(state20.is_ready, "Observation 20 must be ready.")
        self.assertEqual(state20.status, "ACTIVE", "Observation 20 status must be ACTIVE.")
        self.assertIsNotNone(state20.sequence, "Observation 20 sequence must not be None.")
        self.assertEqual(state20.sequence.shape, (20, 20), "Sequence shape must be exactly (20, 20).")
        self.assertEqual(state20.warmup_progress, 1.0, "Warmup progress at step 20 must be 1.0.")

    # 6. Model Inference (TCN-Transformer Multi-Task Prediction)
    def test_06_model_inference(self):
        controller = SumoClosedLoopController(workspace_dir=self.workspace_dir)
        self.assertTrue(controller.model_available, "Hybrid model failed to initialize.")
        self.assertIsNotNone(controller.predictor, "RealTimeTrafficPredictor is None.")

        # Test inference on synthetic sequence [20, 20]
        test_seq = np.random.randn(20, 20).astype(np.float32)
        import torch
        tensor_x = controller.predictor.preprocess_sequence(test_seq)
        with torch.no_grad():
            raw_out = controller.predictor.model(tensor_x)

        preds, conf, is_unc = controller.predictor._decode_predictions(raw_out, idx=0)
        self.assertIn("congestion_level", preds)
        self.assertIn("congestion_score", preds)
        self.assertIn("risk_level", preds)
        self.assertIn("is_approaching", preds)
        self.assertIn("motion_state", preds)
        self.assertIn("has_infraction", preds)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    # 7. Traffic Intelligence (Measured vs Predicted vs Derived)
    def test_07_traffic_intelligence(self):
        policy = ClosedLoopControlPolicy()
        raw_state = {
            "simulation_time": 50.0,
            "step": 50,
            "total_departed": 10,
            "total_arrived": 5,
            "vehicles": {
                "veh1": {"speed": 12.0, "waiting_time": 0.0, "edge_id": "A0A1", "route_edges": ["A0A1", "A1B1"]},
                "veh2": {"speed": 0.0, "waiting_time": 15.0, "edge_id": "A0A1", "route_edges": ["A0A1", "A1B1"]}
            },
            "edges": {
                "A0A1": {"vehicle_count": 2, "mean_speed_kmh": 21.6, "occupancy": 0.1, "queue_length": 1, "waiting_time": 15.0}
            },
            "traffic_lights": {}
        }
        predictions = {
            "veh1": {"congestion_level": {"label": "LOW"}, "congestion_score": 25.0, "risk_level": {"label": "SAFE"}, "is_approaching": False, "confidence": 0.85},
            "veh2": {"congestion_level": {"label": "HIGH"}, "congestion_score": 80.0, "risk_level": {"label": "WARNING"}, "is_approaching": True, "confidence": 0.90}
        }

        intel = policy.compute_traffic_intelligence(raw_state, predictions)
        # Verify strict separation
        self.assertIn("active_vehicles", intel.measured)
        self.assertIn("mean_speed_kmh", intel.measured)
        self.assertIn("total_waiting_time_seconds", intel.measured)
        self.assertEqual(intel.measured["active_vehicles"], 2)
        self.assertEqual(intel.measured["total_queue_vehicles"], 1)

        self.assertIn("predicted_vehicle_count", intel.predicted)
        self.assertIn("average_congestion_score", intel.predicted)
        self.assertEqual(intel.predicted["high_congestion_vehicles"], 1)

        self.assertIn("congestion_index", intel.derived)
        self.assertIn("network_status", intel.derived)
        self.assertIn("throughput_vehicles_per_min", intel.derived)

    # 8. Control Policy Layer (Validation & Whitelist)
    def test_08_control_policy(self):
        policy = ClosedLoopControlPolicy(min_confidence_threshold=0.50)
        self.assertIn("REROUTE_VEHICLE", ALLOWED_ACTIONS)
        self.assertIn("VARIABLE_SPEED_LIMIT", ALLOWED_ACTIONS)
        self.assertIn("RESET_SPEED_LIMIT", ALLOWED_ACTIONS)

        # Valid action
        act = policy._validate_and_build_action(
            action_type="VARIABLE_SPEED_LIMIT",
            target_id="edge_1",
            parameters={"speed_mps": 10.0},
            rationale="Test harmonization",
            confidence=0.85,
            sim_time=10.0
        )
        self.assertIsNotNone(act)
        self.assertTrue(act.is_valid)

        # Unwhitelisted action: must be rejected
        bad_act = policy._validate_and_build_action(
            action_type="INVALID_TELEPORT_COMMAND",
            target_id="edge_1",
            parameters={},
            rationale="Illegal action",
            confidence=0.99,
            sim_time=10.0
        )
        self.assertIsNone(bad_act, "Unwhitelisted action must be rejected.")

        # Out-of-bounds speed: must be rejected
        fast_act = policy._validate_and_build_action(
            action_type="VARIABLE_SPEED_LIMIT",
            target_id="edge_1",
            parameters={"speed_mps": 50.0},
            rationale="Excessive speed",
            confidence=0.90,
            sim_time=10.0
        )
        self.assertIsNone(fast_act, "Out of bounds speed must be rejected.")

    # 9. TraCI Control Actions Dispatch
    def test_09_traci_control_actions(self):
        bridge = TraciBridge(config_file=self.config_file, workspace_dir=self.workspace_dir, label="test_act")
        self.assertTrue(bridge.start())
        for _ in range(5):
            bridge.step()

        # Test lane & edge speed setting
        res_edge = bridge.set_edge_max_speed("A0A1", 11.0)
        self.assertTrue(res_edge, "set_edge_max_speed should succeed.")

        # Test reroute on vehicle if present
        vids = bridge.traci.vehicle.getIDList()
        if vids:
            res_reroute = bridge.reroute_vehicle(vids[0])
            self.assertTrue(res_reroute, "reroute_vehicle should return True.")

        bridge.close()

    # 10. Simulation Continuation (Multi-Step Closed-Loop)
    def test_10_simulation_continuation(self):
        controller = SumoClosedLoopController(workspace_dir=self.workspace_dir, control_enabled=True)
        self.assertTrue(controller.start())
        for step_idx in range(1, 15):
            rec = controller.step()
            self.assertIn("step", rec)
            self.assertEqual(rec["step"], step_idx)
            self.assertIn("latencies_ms", rec)
            self.assertGreater(rec["latencies_ms"]["total"], 0.0)
        controller.stop()

    # 11. Baseline Simulation Results File
    def test_11_baseline_simulation_results(self):
        base_path = os.path.join(self.workspace_dir, "results", "sumo_baseline_results.json")
        self.assertTrue(os.path.exists(base_path), f"Baseline results missing at: {base_path}")
        with open(base_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertFalse(data["control_policy_enabled"])
        t_met = data["traffic_metrics"]
        self.assertIn("total_vehicles_departed", t_met)
        self.assertIn("total_vehicles_completed", t_met)
        self.assertIn("total_travel_time_seconds", t_met)
        self.assertIn("average_travel_time_seconds", t_met)
        self.assertIn("average_speed_kmh", t_met)
        self.assertIn("average_waiting_time_seconds", t_met)
        self.assertIn("throughput_vehs_per_hour", t_met)
        self.assertGreater(t_met["total_vehicles_completed"], 0)

    # 12. Controlled Simulation Results File
    def test_12_controlled_simulation_results(self):
        ctrl_path = os.path.join(self.workspace_dir, "results", "sumo_controlled_results.json")
        self.assertTrue(os.path.exists(ctrl_path), f"Controlled results missing at: {ctrl_path}")
        with open(ctrl_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertTrue(data["control_policy_enabled"])
        t_met = data["traffic_metrics"]
        self.assertIn("total_control_actions_executed", t_met)
        self.assertGreater(t_met["total_control_actions_executed"], 0, "Controlled run must execute control actions.")
        self.assertGreater(t_met["total_vehicles_completed"], 0)

    # 13. Metrics Collection & Objective Comparison File
    def test_13_metrics_collection_and_comparison(self):
        comp_path = os.path.join(self.workspace_dir, "PHASE9_SIMULATION_COMPARISON.md")
        self.assertTrue(os.path.exists(comp_path), "PHASE9_SIMULATION_COMPARISON.md must exist.")
        with open(comp_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Average Travel Time", content)
        self.assertIn("Average Speed", content)
        self.assertIn("Average Waiting Time", content)
        self.assertIn("Throughput Rate", content)
        self.assertIn("Completed Vehicles", content)

    # 14. Clean Shutdown
    def test_14_clean_shutdown(self):
        bridge = TraciBridge(config_file=self.config_file, workspace_dir=self.workspace_dir, label="test_clean_shut")
        bridge.start()
        bridge.step()
        bridge.close()
        self.assertFalse(bridge.is_connected)
        self.assertEqual(bridge.status, "CLOSED")

    # 15. TraCI Failure Handling
    def test_15_traci_failure_handling(self):
        bridge = TraciBridge(config_file=self.config_file, workspace_dir=self.workspace_dir, label="test_fail")
        # Attempting to step when not connected should raise an informative RuntimeError rather than silent corruptions
        with self.assertRaises(RuntimeError):
            bridge.step()

    # 16. Model Failure Handling & Fallback
    def test_16_model_failure_handling(self):
        controller = SumoClosedLoopController(workspace_dir=self.workspace_dir, checkpoint_path="nonexistent_ckpt.pth")
        self.assertFalse(controller.model_available)
        self.assertTrue(controller.start())
        # Stepping should proceed safely despite model being unavailable
        rec = controller.step()
        self.assertEqual(rec["active_inference_vehicles"], 0)
        self.assertIn("simulation_time", rec)
        controller.stop()

    # 17. Authority & Driver Mode Zero-Regression Tests
    def test_17_authority_and_driver_regression(self):
        # 1. Authority status & zones
        r_auth = self.app.get("/api/status")
        self.assertEqual(r_auth.status_code, 200)
        r_zones = self.app.get("/api/zones")
        self.assertEqual(r_zones.status_code, 200)
        r_cams = self.app.get("/api/cameras")
        self.assertEqual(r_cams.status_code, 200)

        # 2. Driver feed & routing
        r_drv = self.app.get("/api/driver/feed?lat=22.7533&lng=75.8937")
        self.assertEqual(r_drv.status_code, 200)
        r_route = self.app.post("/api/routes/evaluate", json={
            "origin": {"lat": 22.7533, "lng": 75.8937},
            "destination": {"lat": 22.7244, "lng": 75.8839}
        })
        self.assertEqual(r_route.status_code, 200)

        # 3. Simulation endpoints
        r_sim = self.app.get("/api/simulation/status")
        self.assertEqual(r_sim.status_code, 200)
        self.assertEqual(r_sim.get_json()["mode"], "SIMULATION DATA")

        r_metrics = self.app.get("/api/simulation/metrics")
        self.assertEqual(r_metrics.status_code, 200)
        self.assertTrue(r_metrics.get_json()["has_baseline"])
        self.assertTrue(r_metrics.get_json()["has_controlled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
