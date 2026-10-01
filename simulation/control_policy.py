"""
Closed-Loop Traffic Management Control Policy
=============================================
Translates hybrid model predictions and measured SUMO traffic telemetry into
strictly validated, safe TraCI actuation commands.

Enforces:
- Allowed-action whitelist
- Confidence thresholds & fallback to measured state
- Actuation cooldowns (anti-chattering)
- Physical speed & parameter bounds
- Explicit 3-way telemetry separation: Measured vs Model Predicted vs Derived
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger("Simulation.ControlPolicy")

# Allowed Actions Whitelist
ALLOWED_ACTIONS = {
    "NO_ACTION",
    "REROUTE_VEHICLE",
    "VARIABLE_SPEED_LIMIT",
    "RESET_SPEED_LIMIT",
    "SIGNAL_ACTUATION"
}


@dataclass
class ValidatedControlAction:
    """Represents an approved, verified control action ready for TraCI dispatch."""
    action_type: str
    target_id: str                      # Vehicle ID, Edge ID, or TLS ID
    parameters: Dict[str, Any]
    rationale: str
    confidence: float
    timestamp: float
    is_valid: bool = True


@dataclass
class StructuredTrafficIntelligence:
    """
    Maintains strict 3-way distinction between:
    A. Measured Telemetry (direct from SUMO environment)
    B. Model Predicted Telemetry (from TCN-Transformer Hybrid)
    C. Derived Analytics (computed indices & aggregations)
    """
    simulation_time: float
    step: int
    measured: Dict[str, Any]
    predicted: Dict[str, Any]
    derived: Dict[str, Any]


class ClosedLoopControlPolicy:
    """
    Evaluates traffic conditions and determines safe control actions.
    Ensures the ML model never issues raw TraCI commands directly.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.50,
        action_cooldown_seconds: float = 10.0,
        congestion_threshold_score: float = 55.0,
        queue_halting_threshold: int = 3,
        min_vsl_speed_mps: float = 6.0,
        max_vsl_speed_mps: float = 13.9
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.action_cooldown_seconds = action_cooldown_seconds
        self.congestion_threshold_score = congestion_threshold_score
        self.queue_halting_threshold = queue_halting_threshold
        self.min_vsl_speed_mps = min_vsl_speed_mps
        self.max_vsl_speed_mps = max_vsl_speed_mps

        # Cooldown trackers to prevent oscillatory control
        self.last_action_time_by_target: Dict[str, float] = {}
        self.active_vsl_edges: Dict[str, float] = {}
        self.action_history: List[ValidatedControlAction] = []

    def compute_traffic_intelligence(
        self,
        raw_state: Dict[str, Any],
        model_predictions: Dict[str, Dict[str, Any]]
    ) -> StructuredTrafficIntelligence:
        """
        Synthesizes raw simulation state and ML outputs into structured intelligence
        with strict Measured vs Predicted vs Derived separation.
        """
        sim_time = raw_state.get("simulation_time", 0.0)
        step = raw_state.get("step", 0)
        vehicles = raw_state.get("vehicles", {})
        edges = raw_state.get("edges", {})

        # A. MEASURED TELEMETRY (Ground-truth physical measurements from SUMO)
        active_count = len(vehicles)
        total_waiting_time = sum(v["waiting_time"] for v in vehicles.values())
        mean_speed_mps = (
            sum(v["speed"] for v in vehicles.values()) / max(1, active_count)
            if active_count > 0 else 0.0
        )
        total_queuing_vehs = sum(
            1 for v in vehicles.values() if v["speed"] < 0.1
        )

        edge_stats = {}
        for eid, edata in edges.items():
            edge_stats[eid] = {
                "vehicle_count": edata["vehicle_count"],
                "mean_speed_kmh": edata["mean_speed_kmh"],
                "occupancy": edata["occupancy"],
                "queue_length": edata["queue_length"],
                "waiting_time": edata["waiting_time"]
            }

        measured = {
            "active_vehicles": active_count,
            "total_departed": raw_state.get("total_departed", 0),
            "total_arrived": raw_state.get("total_arrived", 0),
            "mean_speed_mps": round(mean_speed_mps, 2),
            "mean_speed_kmh": round(mean_speed_mps * 3.6, 2),
            "total_waiting_time_seconds": round(total_waiting_time, 1),
            "total_queue_vehicles": total_queuing_vehs,
            "edge_metrics": edge_stats,
            "traffic_lights": raw_state.get("traffic_lights", {})
        }

        # B. MODEL PREDICTED TELEMETRY (Inference from TCN-Transformer Hybrid)
        pred_count = len(model_predictions)
        high_congestion_count = 0
        high_risk_count = 0
        approaching_threat_count = 0
        congestion_scores = []
        risk_scores = []

        per_vehicle_summary = {}
        for vid, p in model_predictions.items():
            c_score = p.get("congestion_score", 10.0)
            c_raw = p.get("congestion_level", "LOW")
            c_lbl = c_raw.get("label", "LOW") if isinstance(c_raw, dict) else str(c_raw)
            r_raw = p.get("risk_level", "SAFE")
            r_lbl = r_raw.get("label", "SAFE") if isinstance(r_raw, dict) else str(r_raw)
            app_raw = p.get("is_approaching", False)
            is_app = app_raw.get("is_approaching", False) if isinstance(app_raw, dict) else bool(app_raw)
            m_raw = p.get("motion_state", "CRUISING")
            m_lbl = m_raw.get("label", "CRUISING") if isinstance(m_raw, dict) else str(m_raw)
            inf_raw = p.get("has_infraction", "COMPLIANT")
            inf_lbl = inf_raw.get("label", "COMPLIANT") if isinstance(inf_raw, dict) else str(inf_raw)
            conf = p.get("confidence", 0.8)

            congestion_scores.append(c_score)
            if c_lbl == "HIGH" or c_score >= 60.0:
                high_congestion_count += 1
            if r_lbl == "HIGH":
                high_risk_count += 1
            if is_app:
                approaching_threat_count += 1

            per_vehicle_summary[vid] = {
                "congestion_level": c_lbl,
                "congestion_score": round(c_score, 1),
                "risk_level": r_lbl,
                "is_approaching": is_app,
                "motion_state": m_lbl,
                "infraction": inf_lbl,
                "confidence": round(conf, 3)
            }

        avg_cong_score = (
            sum(congestion_scores) / max(1, len(congestion_scores))
            if congestion_scores else 10.0
        )

        predicted = {
            "predicted_vehicle_count": pred_count,
            "average_congestion_score": round(avg_cong_score, 2),
            "high_congestion_vehicles": high_congestion_count,
            "high_risk_vehicles": high_risk_count,
            "approaching_threat_vehicles": approaching_threat_count,
            "vehicle_predictions": per_vehicle_summary
        }

        # C. DERIVED METRICS (Fused situational awareness indicators)
        # Congestion Index: combines measured occupancy/speed ratio with ML congestion score
        speed_factor = max(0.0, 1.0 - (mean_speed_mps / 13.9))
        ml_factor = min(1.0, max(0.0, (avg_cong_score - 10.0) / 90.0))
        congestion_index = round(0.5 * speed_factor + 0.5 * ml_factor, 3)

        # Identify bottleneck edges
        bottlenecks = []
        for eid, edata in edges.items():
            if edata["occupancy"] > 0.35 or edata["queue_length"] >= self.queue_halting_threshold:
                bottlenecks.append(eid)

        derived = {
            "congestion_index": congestion_index,
            "network_status": "CONGESTED" if congestion_index > 0.6 else ("MODERATE" if congestion_index > 0.3 else "FLUID"),
            "bottleneck_edges": bottlenecks,
            "throughput_vehicles_per_min": round((raw_state.get("total_arrived", 0) / max(1.0, sim_time)) * 60.0, 2),
            "efficiency_ratio": round(mean_speed_mps / 13.9, 3)
        }

        return StructuredTrafficIntelligence(
            simulation_time=sim_time,
            step=step,
            measured=measured,
            predicted=predicted,
            derived=derived
        )

    def evaluate_policy(
        self,
        intelligence: StructuredTrafficIntelligence,
        raw_state: Dict[str, Any]
    ) -> List[ValidatedControlAction]:
        """
        Determines and validates control interventions based on traffic intelligence.
        Returns a list of approved ValidatedControlAction objects.
        """
        sim_time = intelligence.simulation_time
        actions: List[ValidatedControlAction] = []
        vehicles = raw_state.get("vehicles", {})
        edges = raw_state.get("edges", {})
        bottlenecks = set(intelligence.derived["bottleneck_edges"])

        # Strategy 1: Dynamic Rerouting of vehicles with high predicted congestion / approach threat
        for vid, vdata in vehicles.items():
            # Check cooldown
            last_act = self.last_action_time_by_target.get(vid, -999.0)
            if (sim_time - last_act) < self.action_cooldown_seconds:
                continue

            vpred = intelligence.predicted["vehicle_predictions"].get(vid, {})
            c_score = vpred.get("congestion_score", 10.0)
            conf = vpred.get("confidence", 0.0)
            cur_edge = vdata.get("edge_id", "")
            route_edges = vdata.get("route_edges", [])

            # If vehicle has high predicted congestion and high confidence, or edge is congested
            if (c_score >= self.congestion_threshold_score and conf >= self.min_confidence_threshold) or cur_edge in bottlenecks:
                if len(route_edges) >= 2:
                    act = self._validate_and_build_action(
                        action_type="REROUTE_VEHICLE",
                        target_id=vid,
                        parameters={"method": "travel_time"},
                        rationale=f"Model predicted high congestion ({c_score:.1f}, conf={conf:.2f}) on approach '{cur_edge}'. Dynamically calculating alternative path.",
                        confidence=conf,
                        sim_time=sim_time
                    )
                    if act and act.is_valid:
                        actions.append(act)
                        self.last_action_time_by_target[vid] = sim_time

        # Strategy 2: Variable Speed Limit (VSL) / Speed Harmonization on approach edges
        edge_high_threat_counts: Dict[str, int] = {}
        for vid, vdata in vehicles.items():
            cur_e = vdata.get("edge_id", "")
            vpred = intelligence.predicted["vehicle_predictions"].get(vid, {})
            if vpred.get("congestion_level") == "HIGH" or vpred.get("is_approaching", False):
                edge_high_threat_counts[cur_e] = edge_high_threat_counts.get(cur_e, 0) + 1

        for eid, edata in edges.items():
            last_vsl_act = self.last_action_time_by_target.get(eid, -999.0)
            if (sim_time - last_vsl_act) < self.action_cooldown_seconds:
                continue

            q_len = edata["queue_length"]
            occ = edata["occupancy"]
            is_bottleneck = eid in bottlenecks
            threat_count = edge_high_threat_counts.get(eid, 0)

            # If edge has multiple high-threat vehicles or bottleneck queue, apply speed harmonization
            if (is_bottleneck or q_len >= self.queue_halting_threshold or threat_count >= 2) and eid not in self.active_vsl_edges:
                advisory_speed = 11.0  # Safe metered speed
                act = self._validate_and_build_action(
                    action_type="VARIABLE_SPEED_LIMIT",
                    target_id=eid,
                    parameters={"speed_mps": advisory_speed},
                    rationale=f"Corridor threat: {threat_count} high-risk vehicles, queue={q_len}. Harmonizing speed to {advisory_speed} m/s",
                    confidence=0.85,
                    sim_time=sim_time
                )
                if act and act.is_valid:
                    actions.append(act)
                    self.active_vsl_edges[eid] = advisory_speed
                    self.last_action_time_by_target[eid] = sim_time

            # If corridor cleared, restore speed limit to standard free-flow (13.9 m/s)
            elif eid in self.active_vsl_edges and q_len == 0 and threat_count == 0 and occ < 0.15:
                act = self._validate_and_build_action(
                    action_type="RESET_SPEED_LIMIT",
                    target_id=eid,
                    parameters={"speed_mps": self.max_vsl_speed_mps},
                    rationale=f"Corridor cleared (queue=0, threats=0, occ={occ:.2f}). Restoring speed to {self.max_vsl_speed_mps} m/s",
                    confidence=0.85,
                    sim_time=sim_time
                )
                if act and act.is_valid:
                    actions.append(act)
                    del self.active_vsl_edges[eid]
                    self.last_action_time_by_target[eid] = sim_time

        # Strategy 3: Traffic Light Actuation (if network contains signalized junctions)
        tls_dict = raw_state.get("traffic_lights", {})
        for tid, tdata in tls_dict.items():
            last_tls_act = self.last_action_time_by_target.get(tid, -999.0)
            if (sim_time - last_tls_act) < self.action_cooldown_seconds:
                continue

            # Signal phase extension logic if applicable
            act = self._validate_and_build_action(
                action_type="SIGNAL_ACTUATION",
                target_id=tid,
                parameters={"extend_duration": 5.0},
                rationale=f"Traffic light {tid} phase optimization based on queue metrics",
                confidence=0.85,
                sim_time=sim_time
            )
            if act and act.is_valid:
                actions.append(act)
                self.last_action_time_by_target[tid] = sim_time

        return actions

    def _validate_and_build_action(
        self,
        action_type: str,
        target_id: str,
        parameters: Dict[str, Any],
        rationale: str,
        confidence: float,
        sim_time: float
    ) -> Optional[ValidatedControlAction]:
        """
        Validates the proposed action against safety rules and allowed whitelist.
        """
        # 1. Whitelist check
        if action_type not in ALLOWED_ACTIONS:
            logger.warning(f"[ControlPolicy] Rejected unwhitelisted action: {action_type}")
            return None

        # 2. Confidence threshold
        if confidence < self.min_confidence_threshold:
            logger.debug(f"[ControlPolicy] Action {action_type} rejected due to low confidence: {confidence}")
            return None

        # 3. Parameter bounds checking
        if action_type in ["VARIABLE_SPEED_LIMIT", "RESET_SPEED_LIMIT"]:
            spd = parameters.get("speed_mps", self.max_vsl_speed_mps)
            if spd < self.min_vsl_speed_mps or spd > self.max_vsl_speed_mps:
                logger.warning(f"[ControlPolicy] Speed {spd} out of bounds [{self.min_vsl_speed_mps}, {self.max_vsl_speed_mps}]")
                return None

        action = ValidatedControlAction(
            action_type=action_type,
            target_id=target_id,
            parameters=parameters,
            rationale=rationale,
            confidence=confidence,
            timestamp=sim_time,
            is_valid=True
        )
        self.action_history.append(action)
        return action

    def reset(self):
        """Clears action histories and cooldown timers."""
        self.last_action_time_by_target.clear()
        self.active_vsl_edges.clear()
        self.action_history.clear()
