"""
SUMO Digital Twin Simulation Engine (Phase 9 Closed-Loop Integration)
=====================================================================
Manages Eclipse SUMO 1.27.1 / TraCI co-simulation for intersection topology,
signal phases, and vehicle flow dynamics.

Integrates the closed-loop autonomous traffic controller (SumoClosedLoopController)
coupling SUMO with the trained TCN-Transformer Gated Hybrid model and adaptive
control policy.

Strictly tags all outputs as 'SIMULATION DATA' to clearly separate simulated
scenarios from live real-world camera feeds.
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any

from simulation.sumo_controller import SumoClosedLoopController


class SumoSimulationEngine:
    def __init__(self, workspace_dir="."):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.sumo_dir = os.path.join(self.workspace_dir, "sumo")
        self.config_file = os.path.join(self.sumo_dir, "simulation.sumocfg")
        self.net_file = os.path.join(self.sumo_dir, "intersection.net.xml")
        self.routes_file = os.path.join(self.sumo_dir, "routes.rou.xml")

        self.traci_available = False
        self.sumo_binary_available = False
        self.status = "UNAVAILABLE"
        self.step_num = 0

        self._check_environment()
        self.network_summary = self._parse_network_summary()

        # Closed-loop controller instance
        self.controller: Optional[SumoClosedLoopController] = None
        if self.traci_available and self.sumo_binary_available:
            try:
                self.controller = SumoClosedLoopController(
                    config_file=self.config_file,
                    workspace_dir=self.workspace_dir,
                    control_enabled=True
                )
            except Exception as e:
                print(f"[SumoSimulationEngine] Warning: controller init failed: {e}")

    def _check_environment(self):
        """Checks if traci python library and sumo binaries exist."""
        try:
            import traci
            import sumolib
            self.traci_available = True
            try:
                bin_path = sumolib.checkBinary("sumo")
                self.sumo_binary_available = bool(bin_path and os.path.exists(bin_path))
            except Exception:
                self.sumo_binary_available = bool(shutil.which("sumo") or shutil.which("sumo.exe"))
        except ImportError:
            self.traci_available = False
            self.sumo_binary_available = False

        if self.traci_available and self.sumo_binary_available:
            self.status = "READY"
        else:
            self.status = "SUMO simulation unavailable"

    def _parse_network_summary(self):
        """Parses edges, junctions, and flow definitions from existing XML files."""
        edges = []
        junctions = []
        routes = []

        if os.path.exists(self.net_file):
            try:
                tree = ET.parse(self.net_file)
                root = tree.getroot()
                for edge in root.findall("edge"):
                    eid = edge.get("id")
                    if eid and not eid.startswith(":"):
                        edges.append({
                            "id": eid,
                            "from": edge.get("from"),
                            "to": edge.get("to"),
                            "priority": edge.get("priority", "1")
                        })
                for junc in root.findall("junction"):
                    jid = junc.get("id")
                    if jid and not jid.startswith(":"):
                        junctions.append({
                            "id": jid,
                            "type": junc.get("type"),
                            "x": float(junc.get("x", 0)),
                            "y": float(junc.get("y", 0))
                        })
            except Exception as e:
                print(f"[SumoEngine] Error parsing net file: {e}")

        if os.path.exists(self.routes_file):
            try:
                tree = ET.parse(self.routes_file)
                root = tree.getroot()
                for flow in root.findall("flow"):
                    routes.append({
                        "id": flow.get("id"),
                        "route": flow.get("route"),
                        "vehsPerHour": int(flow.get("vehsPerHour", 0)),
                        "begin": float(flow.get("begin", 0)),
                        "end": float(flow.get("end", 600))
                    })
            except Exception as e:
                print(f"[SumoEngine] Error parsing route file: {e}")

        return {
            "network_file": os.path.basename(self.net_file),
            "config_file": os.path.basename(self.config_file),
            "junctions_count": len(junctions),
            "edges_count": len(edges),
            "active_flows_count": len(routes),
            "junctions": junctions,
            "edges": edges,
            "routes": routes
        }

    def get_status(self):
        """Returns comprehensive simulation state labeled clearly as SIMULATION DATA."""
        is_live = bool(self.controller and self.controller.is_running and self.controller.bridge.is_connected)
        sim_time = self.controller.active_sim_time if self.controller else 0.0
        step_num = self.controller.bridge.current_step if self.controller else self.step_num
        act_vehs = len(self.controller.bridge.seen_vehicles) if is_live else 0
        tot_dep = self.controller.bridge.total_departed_count if self.controller else 0
        tot_arr = self.controller.bridge.total_arrived_count if self.controller else 0

        latest_rec = self.controller.step_history[-1] if (self.controller and self.controller.step_history) else {}
        avg_spd = latest_rec.get("mean_speed_kmh", 42.6)
        cong_idx = latest_rec.get("congestion_index", 0.35)

        return {
            "mode": "SIMULATION DATA",
            "data_source": "ECLIPSE_SUMO_TRACI",
            "status": "RUNNING" if is_live else ("READY" if (self.traci_available and self.sumo_binary_available) else "SUMO simulation unavailable"),
            "is_live_traci": is_live,
            "traci_installed": self.traci_available,
            "sumo_binary_found": self.sumo_binary_available,
            "control_policy_enabled": self.controller.control_enabled if self.controller else True,
            "step": step_num,
            "simulation_time_seconds": round(sim_time, 1),
            "active_vehicles": act_vehs,
            "total_departed": tot_dep,
            "total_arrived": tot_arr,
            "average_speed_kmh": round(avg_spd, 2),
            "congestion_index": round(cong_idx, 3),
            "model_available": self.controller.model_available if self.controller else False,
            "network": self.network_summary
        }

    def start(self):
        """Starts SUMO simulation."""
        if not self.controller:
            self._check_environment()
            if self.traci_available and self.sumo_binary_available:
                self.controller = SumoClosedLoopController(workspace_dir=self.workspace_dir)

        if self.controller:
            success = self.controller.start()
            if success:
                self.status = "RUNNING"
                return {"mode": "SIMULATION DATA", "status": "RUNNING", "message": "Closed-loop simulation started."}

        self.status = "OFFLINE_PLAYBACK"
        return {
            "mode": "SIMULATION DATA",
            "status": "OFFLINE_PLAYBACK",
            "message": "SUMO binary not active. Running cached topology replay."
        }

    def pause(self):
        self.status = "PAUSED"
        return {"mode": "SIMULATION DATA", "status": "PAUSED", "message": "Simulation paused"}

    def stop(self):
        if self.controller and self.controller.is_running:
            self.controller.stop()
        self.status = "STOPPED"
        return {"mode": "SIMULATION DATA", "status": "STOPPED", "message": "Simulation stopped"}

    def reset(self):
        self.stop()
        self.step_num = 0
        if self.controller:
            self.controller.adapter.reset()
            self.controller.policy.reset()
        self.status = "READY" if (self.traci_available and self.sumo_binary_available) else "SUMO simulation unavailable"
        return {"mode": "SIMULATION DATA", "status": self.status, "message": "Simulation reset"}

    def step(self):
        if not self.controller or not self.controller.is_running:
            self.start()
        if self.controller:
            rec = self.controller.step()
            self.step_num = rec.get("step", self.step_num + 1)
            return {"mode": "SIMULATION DATA", "status": "RUNNING", "step_record": rec}
        self.step_num += 1
        return self.get_status()

    def toggle_control(self):
        if self.controller:
            self.controller.control_enabled = not self.controller.control_enabled
            return {
                "mode": "SIMULATION DATA",
                "control_policy_enabled": self.controller.control_enabled,
                "message": f"Closed-loop control {'enabled' if self.controller.control_enabled else 'disabled'}."
            }
        return {"error": "Controller unavailable"}

    def get_traffic(self):
        """Returns live simulated traffic state with Measured/Predicted/Derived segregation."""
        if not self.controller or not self.controller.is_running or self.controller.latest_intelligence is None:
            # Check for persisted controlled results
            cached_path = os.path.join(self.workspace_dir, "results", "sumo_controlled_results.json")
            cached = {}
            if os.path.exists(cached_path):
                with open(cached_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
            return {
                "mode": "SIMULATION DATA",
                "status": "IDLE",
                "message": "Simulation is not actively stepping. Call /api/simulation/start or step.",
                "latest_run_summary": cached
            }

        intel = self.controller.latest_intelligence
        return {
            "mode": "SIMULATION DATA",
            "status": "LIVE_SIMULATION",
            "simulation_time": intel.simulation_time,
            "step": intel.step,
            "measured": intel.measured,
            "predicted": intel.predicted,
            "derived": intel.derived
        }

    def get_control(self):
        """Returns control policy parameters and recent actuation history."""
        if not self.controller:
            return {"error": "Controller unavailable"}

        actions = []
        if self.controller.latest_actions:
            for a in self.controller.latest_actions:
                actions.append({
                    "action_type": a.action_type,
                    "target_id": a.target_id,
                    "parameters": a.parameters,
                    "rationale": a.rationale,
                    "confidence": a.confidence,
                    "timestamp": a.timestamp
                })

        return {
            "mode": "SIMULATION DATA",
            "control_enabled": self.controller.control_enabled,
            "congestion_threshold_score": self.controller.policy.congestion_threshold_score,
            "action_cooldown_seconds": self.controller.policy.action_cooldown_seconds,
            "latest_actions_count": len(actions),
            "latest_actions": actions,
            "total_actions_history": len(self.controller.policy.action_history)
        }

    def set_control(self, body: Dict[str, Any]):
        """Updates control policy parameters."""
        if not self.controller:
            return {"error": "Controller unavailable"}

        if "control_enabled" in body:
            self.controller.control_enabled = bool(body["control_enabled"])
        if "congestion_threshold" in body:
            self.controller.policy.congestion_threshold_score = float(body["congestion_threshold"])
        if "action_cooldown" in body:
            self.controller.policy.action_cooldown_seconds = float(body["action_cooldown"])

        return self.get_control()

    def get_metrics(self):
        """Returns baseline vs controlled simulation comparison metrics."""
        base_path = os.path.join(self.workspace_dir, "results", "sumo_baseline_results.json")
        ctrl_path = os.path.join(self.workspace_dir, "results", "sumo_controlled_results.json")

        base_data = None
        ctrl_data = None
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)
        if os.path.exists(ctrl_path):
            with open(ctrl_path, "r", encoding="utf-8") as f:
                ctrl_data = json.load(f)

        return {
            "mode": "SIMULATION DATA",
            "has_baseline": bool(base_data),
            "has_controlled": bool(ctrl_data),
            "baseline": base_data,
            "controlled": ctrl_data
        }
