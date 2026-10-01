"""
SUMO Closed-Loop Traffic Intelligence Controller
=================================================
Top-level orchestrator executing the autonomous closed-loop traffic simulation:
SUMO -> TraCI -> State Extraction -> Feature Engineering -> Temporal Sequence (20x20)
     -> TCN-Transformer Hybrid Model -> Traffic Intelligence -> Control Policy
     -> Validated TraCI Action -> SUMO -> Repeat

Provides:
- Synchronous baseline evaluation (uncontrolled)
- Synchronous controlled evaluation (active adaptive intelligence)
- Single-step interactive stepping for live API / UI dashboard
- High-precision telemetry profiling & latency benchmarks
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simulation.traci_bridge import TraciBridge
from simulation.traffic_state_adapter import TrafficStateAdapter, VehicleSequenceState
from simulation.control_policy import ClosedLoopControlPolicy, StructuredTrafficIntelligence, ValidatedControlAction
from ml.hybrid_traffic.realtime_inference import RealTimeTrafficPredictor

logger = logging.getLogger("Simulation.ClosedLoopController")


class SumoClosedLoopController:
    """
    Closed-loop coordinator executing observation, inference, decision-making,
    and actuation between SUMO and the hybrid intelligence architecture.
    """

    def __init__(
        self,
        config_file: str = "sumo/simulation.sumocfg",
        workspace_dir: str = ".",
        checkpoint_path: str = "checkpoints/best_model.pth",
        scaler_path: str = "checkpoints/feature_scaler.joblib",
        model_config_path: str = "checkpoints/best_model_config.json",
        device: str = "cpu",
        control_enabled: bool = True,
        gui: bool = False
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.config_file = config_file
        self.control_enabled = control_enabled
        self.gui = gui

        # Subsystems
        self.bridge = TraciBridge(
            config_file=self.config_file,
            workspace_dir=self.workspace_dir,
            gui=self.gui
        )
        self.adapter = TrafficStateAdapter()
        self.policy = ClosedLoopControlPolicy()

        # Hybrid Model
        self.model_available = False
        self.predictor = None
        self._init_hybrid_model(checkpoint_path, scaler_path, model_config_path, device)

        # Performance profiling telemetry (rolling arrays)
        self.latencies = {
            "traci_extraction_ms": [],
            "feature_generation_ms": [],
            "model_inference_ms": [],
            "intelligence_engine_ms": [],
            "control_policy_ms": [],
            "actuation_ms": [],
            "total_step_ms": []
        }

        # Step record history
        self.step_history: List[Dict[str, Any]] = []
        self.latest_intelligence: Optional[StructuredTrafficIntelligence] = None
        self.latest_actions: List[ValidatedControlAction] = []
        self.active_sim_time = 0.0
        self.is_running = False

    def _init_hybrid_model(
        self,
        ckpt_path: str,
        scaler_path: str,
        cfg_path: str,
        device: str
    ):
        """Loads the trained TCN-Transformer Gated Hybrid model."""
        abs_ckpt = os.path.join(self.workspace_dir, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
        abs_scaler = os.path.join(self.workspace_dir, scaler_path) if not os.path.isabs(scaler_path) else scaler_path
        abs_cfg = os.path.join(self.workspace_dir, cfg_path) if not os.path.isabs(cfg_path) else cfg_path

        try:
            self.predictor = RealTimeTrafficPredictor(
                checkpoint_path=abs_ckpt,
                scaler_path=abs_scaler,
                config_path=abs_cfg,
                device=device
            )
            self.model_available = True
            logger.info("[ClosedLoopController] Loaded TCN-Transformer Hybrid model successfully.")
        except Exception as e:
            logger.error(f"[ClosedLoopController] Model loading failed: {e}. Falling back to heuristic mode.")
            self.model_available = False
            self.predictor = None

    def start(self, additional_args: Optional[List[str]] = None) -> bool:
        """Starts SUMO and initializes state buffers."""
        success = self.bridge.start(additional_args=additional_args)
        if success:
            self.adapter.reset()
            self.policy.reset()
            self.step_history.clear()
            for k in self.latencies:
                self.latencies[k].clear()
            self.is_running = True
            self.active_sim_time = 0.0
        return success

    def step(self) -> Dict[str, Any]:
        """
        Executes one complete closed-loop cycle:
        1. SUMO simulation step & raw state extraction
        2. Feature mapping & temporal sequence accumulation
        3. Hybrid model multi-task inference on ready sequences
        4. Traffic intelligence synthesis (Measured vs Predicted vs Derived)
        5. Control policy evaluation
        6. TraCI actuation dispatch
        7. Latency and telemetry recording
        """
        if not self.is_running or not self.bridge.is_connected:
            return {"status": "STOPPED", "message": "Simulation is not running"}

        t_total_start = time.perf_counter()

        # Step 1: TraCI extraction
        t_ext_start = time.perf_counter()
        raw_state = self.bridge.step()
        t_ext_end = time.perf_counter()

        if raw_state.get("status") == "TERMINATED":
            self.is_running = False
            return {"status": "TERMINATED", "step": self.bridge.current_step, "simulation_time": self.bridge.current_sim_time}

        sim_time = raw_state["simulation_time"]
        step_num = raw_state["step"]
        self.active_sim_time = sim_time
        vehicles = raw_state["vehicles"]

        # Step 2: Feature mapping & sequence buffering
        t_feat_start = time.perf_counter()
        active_vids = set(vehicles.keys())
        self.adapter.prune_departed(active_vids)

        ready_sequences = {}
        warming_up_count = 0
        active_count = 0

        for vid, vdata in vehicles.items():
            seq_state: VehicleSequenceState = self.adapter.update_vehicle(
                obs=vdata,
                step_num=step_num,
                sim_time=sim_time,
                dt=self.bridge.step_length
            )
            if seq_state.is_ready:
                ready_sequences[vid] = seq_state.sequence
                active_count += 1
            else:
                warming_up_count += 1
        t_feat_end = time.perf_counter()

        # Step 3: Hybrid model inference
        t_infer_start = time.perf_counter()
        predictions = {}
        if self.model_available and self.predictor and len(ready_sequences) > 0:
            try:
                import torch
                vids = list(ready_sequences.keys())
                seq_list = [ready_sequences[v] for v in vids]
                batch_arr = np.stack(seq_list, axis=0)  # Shape (B, 20, 20)
                tensor_batch = self.predictor.preprocess_sequence(batch_arr)

                with torch.no_grad():
                    raw_outputs = self.predictor.model(tensor_batch)

                for b_idx, vid in enumerate(vids):
                    preds, conf, is_unc = self.predictor._decode_predictions(raw_outputs, idx=b_idx)
                    preds["confidence"] = conf
                    preds["is_uncertain"] = is_unc
                    predictions[vid] = preds
            except Exception as e:
                logger.warning(f"[ClosedLoopController] Inference error: {e}. Falling back to safe measured state.")
                predictions = {}
        t_infer_end = time.perf_counter()

        # Step 4: Traffic intelligence engine
        t_intel_start = time.perf_counter()
        intelligence = self.policy.compute_traffic_intelligence(
            raw_state=raw_state,
            model_predictions=predictions
        )
        self.latest_intelligence = intelligence
        t_intel_end = time.perf_counter()

        # Step 5: Control policy evaluation
        t_policy_start = time.perf_counter()
        actions = []
        if self.control_enabled:
            actions = self.policy.evaluate_policy(
                intelligence=intelligence,
                raw_state=raw_state
            )
        self.latest_actions = actions
        t_policy_end = time.perf_counter()

        # Step 6: TraCI actuation
        t_act_start = time.perf_counter()
        executed_actions = []
        for act in actions:
            success = self._dispatch_action(act)
            if success:
                executed_actions.append(act)
        t_act_end = time.perf_counter()

        t_total_end = time.perf_counter()

        # Compute latencies in milliseconds
        ext_ms = (t_ext_end - t_ext_start) * 1000.0
        feat_ms = (t_feat_end - t_feat_start) * 1000.0
        infer_ms = (t_infer_end - t_infer_start) * 1000.0
        intel_ms = (t_intel_end - t_intel_start) * 1000.0
        pol_ms = (t_policy_end - t_policy_start) * 1000.0
        act_ms = (t_act_end - t_act_start) * 1000.0
        tot_ms = (t_total_end - t_total_start) * 1000.0

        self.latencies["traci_extraction_ms"].append(ext_ms)
        self.latencies["feature_generation_ms"].append(feat_ms)
        self.latencies["model_inference_ms"].append(infer_ms)
        self.latencies["intelligence_engine_ms"].append(intel_ms)
        self.latencies["control_policy_ms"].append(pol_ms)
        self.latencies["actuation_ms"].append(act_ms)
        self.latencies["total_step_ms"].append(tot_ms)

        step_record = {
            "step": step_num,
            "simulation_time": sim_time,
            "active_vehicles": len(vehicles),
            "warming_up_vehicles": warming_up_count,
            "active_inference_vehicles": active_count,
            "mean_speed_kmh": intelligence.measured["mean_speed_kmh"],
            "total_waiting_time": intelligence.measured["total_waiting_time_seconds"],
            "queue_length": intelligence.measured["total_queue_vehicles"],
            "congestion_index": intelligence.derived["congestion_index"],
            "actions_executed": len(executed_actions),
            "latencies_ms": {
                "extraction": round(ext_ms, 3),
                "features": round(feat_ms, 3),
                "inference": round(infer_ms, 3),
                "intelligence": round(intel_ms, 3),
                "policy": round(pol_ms, 3),
                "actuation": round(act_ms, 3),
                "total": round(tot_ms, 3)
            }
        }
        self.step_history.append(step_record)
        return step_record

    def _dispatch_action(self, action: ValidatedControlAction) -> bool:
        """Executes a validated control action via TraCI bridge."""
        if not action.is_valid:
            return False

        try:
            if action.action_type == "REROUTE_VEHICLE":
                return self.bridge.reroute_vehicle(action.target_id)
            elif action.action_type == "VARIABLE_SPEED_LIMIT":
                spd = action.parameters.get("speed_mps", 10.0)
                return self.bridge.set_edge_max_speed(action.target_id, spd)
            elif action.action_type == "RESET_SPEED_LIMIT":
                spd = action.parameters.get("speed_mps", 13.9)
                return self.bridge.set_edge_max_speed(action.target_id, spd)
            elif action.action_type == "SIGNAL_ACTUATION":
                # If phase index is passed
                if "phase_index" in action.parameters:
                    return self.bridge.set_traffic_light_phase(action.target_id, action.parameters["phase_index"])
                return True
            return False
        except Exception as e:
            logger.warning(f"[ClosedLoopController] Action dispatch error for {action.action_type}: {e}")
            return False

    def run_simulation(
        self,
        duration: int = 600,
        control_enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Runs a complete scenario for `duration` seconds and calculates final metrics.
        """
        if control_enabled is not None:
            self.control_enabled = control_enabled

        started = self.start()
        if not started:
            raise RuntimeError("Failed to start SUMO simulation.")

        # Run loop until target duration reached or simulation ends
        while self.is_running and self.bridge.current_sim_time < duration:
            rec = self.step()
            if rec.get("status") == "TERMINATED":
                break

        final_summary = self.compile_simulation_results()
        self.stop()
        return final_summary

    def compile_simulation_results(self) -> Dict[str, Any]:
        """Aggregates comprehensive physical traffic performance metrics."""
        tot_departed = self.bridge.total_departed_count
        tot_arrived = self.bridge.total_arrived_count
        final_sim_time = self.bridge.current_sim_time
        completed_tts = self.bridge.completed_travel_times

        tot_travel_time = float(sum(completed_tts)) if completed_tts else 0.0
        avg_travel_time = float(np.mean(completed_tts)) if completed_tts else 0.0

        speeds = [s["mean_speed_kmh"] for s in self.step_history if s["active_vehicles"] > 0]
        waits = [s["total_waiting_time"] for s in self.step_history]
        cong_indices = [s["congestion_index"] for s in self.step_history]
        queues = [s.get("queue_length", s.get("total_queue_vehicles", 0)) for s in self.step_history]

        avg_speed_kmh = float(np.mean(speeds)) if speeds else 0.0
        final_waiting_time = float(waits[-1]) if waits else 0.0
        avg_waiting_time = float(np.mean(waits)) if waits else 0.0
        avg_congestion = float(np.mean(cong_indices)) if cong_indices else 0.0
        avg_queue_len = float(np.mean(queues)) if queues else 0.0
        max_queue_len = int(np.max(queues)) if queues else 0

        # Latency statistics
        tot_lats = self.latencies["total_step_ms"]
        avg_total_lat = float(np.mean(tot_lats)) if tot_lats else 0.0
        max_total_lat = float(np.max(tot_lats)) if tot_lats else 0.0
        effective_fps = round(1000.0 / max(1e-3, avg_total_lat), 1) if avg_total_lat > 0 else 0.0

        infer_lats = [x for x in self.latencies["model_inference_ms"] if x > 0.0]
        avg_infer_lat = float(np.mean(infer_lats)) if infer_lats else 0.0

        throughput_vph = round((tot_arrived / max(1.0, final_sim_time)) * 3600.0, 1)

        return {
            "control_policy_enabled": self.control_enabled,
            "simulation_duration_seconds": round(final_sim_time, 1),
            "total_steps_executed": len(self.step_history),
            "traffic_metrics": {
                "total_vehicles_departed": tot_departed,
                "total_vehicles_completed": tot_arrived,
                "throughput_vehs_per_hour": throughput_vph,
                "total_travel_time_seconds": round(tot_travel_time, 2),
                "average_travel_time_seconds": round(avg_travel_time, 2),
                "average_speed_kmh": round(avg_speed_kmh, 2),
                "average_waiting_time_seconds": round(avg_waiting_time, 2),
                "final_cumulative_waiting_time": round(final_waiting_time, 1),
                "average_queue_length": round(avg_queue_len, 2),
                "max_queue_length": max_queue_len,
                "average_congestion_index": round(avg_congestion, 3),
                "total_control_actions_executed": sum(s["actions_executed"] for s in self.step_history)
            },
            "latency_metrics": {
                "average_total_step_ms": round(avg_total_lat, 2),
                "max_total_step_ms": round(max_total_lat, 2),
                "effective_simulation_fps": effective_fps,
                "average_model_inference_ms": round(avg_infer_lat, 2),
                "average_traci_extraction_ms": round(float(np.mean(self.latencies["traci_extraction_ms"])), 2) if self.latencies["traci_extraction_ms"] else 0.0,
                "average_feature_generation_ms": round(float(np.mean(self.latencies["feature_generation_ms"])), 2) if self.latencies["feature_generation_ms"] else 0.0,
                "average_intelligence_engine_ms": round(float(np.mean(self.latencies["intelligence_engine_ms"])), 2) if self.latencies["intelligence_engine_ms"] else 0.0,
                "average_control_policy_ms": round(float(np.mean(self.latencies["control_policy_ms"])), 2) if self.latencies["control_policy_ms"] else 0.0
            }
        }

    def stop(self):
        """Terminates simulation and releases TraCI resources."""
        self.is_running = False
        self.bridge.close()
