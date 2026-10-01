"""
Traffic Intelligence Aggregation Engine
=======================================

Aggregates vehicle-level hybrid model predictions alongside deterministic traffic
measurements across four hierarchical levels:
1. Vehicle-Level Intelligence
2. Lane-Level Intelligence
3. Zone-Level Intelligence (A. Measured Values, B. Model Predictions, C. Derived Intelligence)
4. Overall System Traffic Intelligence

Core Safety Principle:
- Measured ground truth (counts, velocities, positions) is never overwritten.
- Model predictions enhance deterministic calculations via adaptive confidence-weighted fusion.
- Insufficient history and uncertain outputs gracefully fall back to deterministic baselines.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.hybrid_traffic.config import ZONE_TO_LANE_MAP

logger = logging.getLogger("TrafficIntelligence.Aggregator")


class TrafficIntelligenceAggregator:
    """
    Hierarchical aggregation engine fusing deterministic telemetry with
    multi-task TCN-Transformer hybrid model predictions.
    """

    def __init__(
        self,
        zone_descriptions: Optional[Dict[str, str]] = None,
        zone_to_lane_map: Optional[Dict[str, str]] = None,
        base_model_weight: float = 0.40,
    ):
        self.zone_descriptions = zone_descriptions or {
            "ZONE 1": "Main Approach Corridor (North Inflow)",
            "ZONE 2": "Eastbound Exit / Turning Lane",
            "ZONE 3": "Intersection Core Junction Box",
            "ZONE 4": "Southbound Queue Area",
            "ZONE 5": "Westbound Inflow Lane",
            "ZONE 6": "Southeast Exit Lane",
        }
        self.zone_to_lane = zone_to_lane_map or ZONE_TO_LANE_MAP
        self.base_model_weight = base_model_weight

    def aggregate(
        self,
        frame_num: int,
        time_seconds: float,
        measured_detections: List[Dict[str, Any]],
        model_predictions: List[Dict[str, Any]],
        deterministic_zones: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Combines frame observations and model predictions into full-spectrum intelligence.
        
        Args:
            frame_num: Video frame index.
            time_seconds: Video timestamp in seconds.
            measured_detections: Raw detection list with positions, classes, zones.
            model_predictions: Output list from RealTimeTrafficPredictor.
            deterministic_zones: Optional existing zone telemetry from TrafficIntelligenceEngine.
            
        Returns:
            Dict containing vehicle_level, lane_level, zone_level, and system_level intelligence.
        """
        pred_map = {p["track_id"]: p for p in model_predictions if "track_id" in p}

        # ---------------------------------------------------------------------
        # 1. VEHICLE-LEVEL INTELLIGENCE
        # ---------------------------------------------------------------------
        vehicle_level = {}
        for det in measured_detections:
            tid = det["track_id"]
            p = pred_map.get(tid, {})

            v_entry = {
                "track_id": tid,
                "class_id": det.get("class_id", det.get("class", 2)),
                "vehicle_type": det.get("vehicle_type", "vehicle"),
                "bbox": [det.get("x1", 0.0), det.get("y1", 0.0), det.get("x2", 0.0), det.get("y2", 0.0)],
                "contact_point": [det.get("bottom_x", det.get("center_x", 0.0)), det.get("bottom_y", det.get("y2", 0.0))],
                "zone_id": det.get("zone", p.get("zone_id", "UNKNOWN")),
                "lane_id": det.get("lane_id", p.get("lane_id", "UNKNOWN")),
                "measured": {
                    "center_x": det.get("center_x", 0.0),
                    "center_y": det.get("center_y", 0.0),
                    "speed_px_per_sec": det.get("speed_px_per_sec", 0.0),
                },
                "prediction": {
                    "prediction_ready": p.get("prediction_ready", False),
                    "status": p.get("status", "WARMING_UP"),
                    "warmup_progress": p.get("warmup_progress", 0.0),
                    "confidence": p.get("confidence", 0.0),
                    "is_uncertain": p.get("is_uncertain", True),
                    "safety_advisory": p.get("safety_advisory", "Trajectory warming up"),
                    "details": p.get("predictions", None),
                }
            }
            vehicle_level[tid] = v_entry

        # ---------------------------------------------------------------------
        # 2. LANE-LEVEL INTELLIGENCE
        # ---------------------------------------------------------------------
        lanes_data: Dict[str, Dict[str, Any]] = {}
        for v in vehicle_level.values():
            lid = v["lane_id"]
            if lid not in lanes_data:
                lanes_data[lid] = {
                    "lane_id": lid,
                    "vehicle_count": 0,
                    "speeds": [],
                    "congestion_levels": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
                    "risk_levels": {"SAFE": 0, "WARNING": 0, "HIGH": 0},
                    "approaching_count": 0,
                    "infraction_count": 0,
                }
            lanes_data[lid]["vehicle_count"] += 1
            spd = v["measured"]["speed_px_per_sec"]
            if spd > 0:
                lanes_data[lid]["speeds"].append(spd)

            preds = v["prediction"]["details"]
            if preds and v["prediction"]["prediction_ready"]:
                # Congestion level
                clevel = preds.get("congestion_level", {}).get("label")
                if clevel in lanes_data[lid]["congestion_levels"]:
                    lanes_data[lid]["congestion_levels"][clevel] += 1
                # Risk level
                rlevel = preds.get("risk_level", {}).get("label")
                if rlevel in lanes_data[lid]["risk_levels"]:
                    lanes_data[lid]["risk_levels"][rlevel] += 1
                # Approaching
                if preds.get("is_approaching", {}).get("is_approaching", False):
                    lanes_data[lid]["approaching_count"] += 1
                # Infraction
                if preds.get("has_infraction", {}).get("violation_detected", False):
                    lanes_data[lid]["infraction_count"] += 1

        lane_level = {}
        for lid, linfo in lanes_data.items():
            speeds = linfo["speeds"]
            lane_level[lid] = {
                "lane_id": lid,
                "vehicle_count": linfo["vehicle_count"],
                "average_speed_px_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "congestion_level_distribution": linfo["congestion_levels"],
                "risk_level_distribution": linfo["risk_levels"],
                "approaching_vehicles_count": linfo["approaching_count"],
                "infractions_detected_count": linfo["infraction_count"],
            }

        # ---------------------------------------------------------------------
        # 3. ZONE-LEVEL INTELLIGENCE (SEPARATE A, B, C)
        # ---------------------------------------------------------------------
        # Ensure all standard zones (ZONE 1 .. ZONE 6) exist in output
        standard_zones = ["ZONE 1", "ZONE 2", "ZONE 3", "ZONE 4", "ZONE 5", "ZONE 6"]
        zone_level = {}

        for z_name in standard_zones:
            # Gather vehicles in this zone
            z_vehs = [v for v in vehicle_level.values() if v["zone_id"] == z_name]
            det_z_info = deterministic_zones.get(z_name, {}) if deterministic_zones else {}

            # --- A. MEASURED VALUES ---
            z_count = len(z_vehs)
            z_speeds = [v["measured"]["speed_px_per_sec"] for v in z_vehs if v["measured"]["speed_px_per_sec"] > 0]
            avg_measured_speed = round(float(np.mean(z_speeds)), 2) if z_speeds else 0.0

            modal_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for v in z_vehs:
                vtype = v["vehicle_type"].lower()
                if vtype in modal_counts:
                    modal_counts[vtype] += 1
                else:
                    modal_counts["car"] += 1

            # Deterministic density score: (count / 200) * 100 capped at 100
            density_score = round(min(100.0, (z_count / 200.0) * 100.0), 2)
            det_level = det_z_info.get("level", "LOW" if z_count <= 2 else ("MEDIUM" if z_count <= 5 else "HIGH"))
            det_trend = det_z_info.get("trend", "STABLE")
            det_avg_count = det_z_info.get("average_count", float(z_count))
            det_sustained = det_z_info.get("sustained_seconds", 0.0)

            measured_values = {
                "vehicle_count": z_count,
                "average_speed_px_per_sec": avg_measured_speed,
                "modal_breakdown": modal_counts,
                "density_score": density_score,
                "deterministic_level": det_level,
                "deterministic_trend": det_trend,
                "deterministic_average_count": det_avg_count,
                "sustained_high_seconds": det_sustained,
            }

            # --- B. MODEL PREDICTIONS ---
            ready_vehs = [v for v in z_vehs if v["prediction"]["prediction_ready"]]
            ready_count = len(ready_vehs)
            warmup_count = z_count - ready_count

            cong_scores = []
            confidences = []
            uncertain_count = 0
            cong_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            risk_levels = {"SAFE": 0, "WARNING": 0, "HIGH": 0}
            motion_states = {"STOPPED": 0, "ACCELERATING": 0, "DECELERATING": 0, "CRUISING": 0}
            maneuvers = {"STATIONARY": 0, "TURNING_LEFT": 0, "TURNING_RIGHT": 0, "STRAIGHT": 0}
            approaching_count = 0
            approach_scores = []
            infraction_count = 0

            for v in ready_vehs:
                p_details = v["prediction"]["details"]
                if p_details:
                    cong_scores.append(p_details.get("congestion_score", 30.0))
                    cl = p_details.get("congestion_level", {}).get("label")
                    if cl in cong_levels:
                        cong_levels[cl] += 1

                    rl = p_details.get("risk_level", {}).get("label")
                    if rl in risk_levels:
                        risk_levels[rl] += 1

                    ms = p_details.get("motion_state", {}).get("label")
                    if ms in motion_states:
                        motion_states[ms] += 1

                    mn = p_details.get("maneuver_type", {}).get("label")
                    if mn in maneuvers:
                        maneuvers[mn] += 1

                    if p_details.get("is_approaching", {}).get("is_approaching", False):
                        approaching_count += 1
                        approach_scores.append(p_details.get("approach_threat_score", 0.0))

                    if p_details.get("has_infraction", {}).get("violation_detected", False):
                        infraction_count += 1

                confidences.append(v["prediction"]["confidence"])
                if v["prediction"]["is_uncertain"]:
                    uncertain_count += 1

            mean_pred_cong_score = round(float(np.mean(cong_scores)), 2) if cong_scores else None
            mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
            mean_app_threat = round(float(np.mean(approach_scores)), 2) if approach_scores else 0.0

            model_predictions_summary = {
                "predictions_ready_count": ready_count,
                "warming_up_count": warmup_count,
                "mean_predicted_congestion_score": mean_pred_cong_score,
                "predicted_congestion_distribution": cong_levels,
                "predicted_risk_distribution": risk_levels,
                "predicted_motion_distribution": motion_states,
                "predicted_maneuver_distribution": maneuvers,
                "approaching_vehicles_count": approaching_count,
                "mean_approach_threat_score": mean_app_threat,
                "predicted_infractions_count": infraction_count,
                "average_model_confidence": mean_conf,
                "uncertain_predictions_count": uncertain_count,
            }

            # --- C. DERIVED INTELLIGENCE ---
            # Fused Congestion Score calculation:
            # Deterministic base: mapping det_level to score (LOW: 20, MEDIUM: 50, HIGH: 80) adjusted by count
            det_base_score = float(min(100.0, max(10.0, density_score * 0.7 + (15.0 if det_level == "LOW" else (45.0 if det_level == "MEDIUM" else 75.0)) * 0.3)))

            if mean_pred_cong_score is not None and ready_count > 0:
                # Dynamic adaptive weight: discount model weight if confidence is low or many tracks warming up
                ready_ratio = ready_count / max(1, z_count)
                model_w = self.base_model_weight * mean_conf * ready_ratio
                model_w = min(0.50, max(0.05, model_w))
                det_w = 1.0 - model_w
                fused_cong_score = round(det_w * det_base_score + model_w * mean_pred_cong_score, 2)
            else:
                fused_cong_score = round(det_base_score, 2)

            # Classify fused level
            if fused_cong_score < 25.0:
                fused_level = "LOW"
            elif fused_cong_score < 50.0:
                fused_level = "MODERATE"
            elif fused_cong_score < 75.0:
                fused_level = "HIGH"
            else:
                fused_level = "SEVERE"

            # Derive actionable traffic decision
            if fused_level in ["HIGH", "SEVERE"] and det_trend == "INCREASING":
                action = "Trigger Green Phase Extension (+15s) — Influx clearing"
                safety_status = "CRITICAL"
            elif fused_level in ["HIGH", "SEVERE"]:
                action = "Hold Green Split — Sustained queue clearance"
                safety_status = "WARNING"
            elif approaching_count >= 2 or risk_levels["HIGH"] >= 1:
                action = "Advisory: Elevated approach velocity detected"
                safety_status = "ADVISORY"
            else:
                action = "Maintain Nominal Cycle Split"
                safety_status = "NOMINAL"

            derived_intelligence = {
                "fused_congestion_score": fused_cong_score,
                "fused_congestion_level": fused_level,
                "deterministic_base_score": round(det_base_score, 2),
                "model_predicted_score": mean_pred_cong_score,
                "safety_status": safety_status,
                "recommended_action": action,
            }

            zone_level[z_name] = {
                "zone_id": z_name,
                "description": self.zone_descriptions.get(z_name, ""),
                "measured_values": measured_values,
                "model_predictions": model_predictions_summary,
                "derived_intelligence": derived_intelligence,
            }

        # ---------------------------------------------------------------------
        # 4. OVERALL SYSTEM-LEVEL INTELLIGENCE
        # ---------------------------------------------------------------------
        total_active = len(measured_detections)
        ready_total = sum(1 for v in vehicle_level.values() if v["prediction"]["prediction_ready"])
        warmup_total = total_active - ready_total

        # Average fused score
        all_fused_scores = [z["derived_intelligence"]["fused_congestion_score"] for z in zone_level.values()]
        sys_fused_score = round(float(np.mean(all_fused_scores)), 2) if all_fused_scores else 0.0

        # Elect priority zone (highest fused score with high trend tiebreaker)
        trend_weight = {"INCREASING": 15, "STABLE": 5, "DECREASING": 0}
        priority_zone = max(
            zone_level.keys(),
            key=lambda z: (
                zone_level[z]["derived_intelligence"]["fused_congestion_score"]
                + trend_weight.get(zone_level[z]["measured_values"]["deterministic_trend"], 0)
            )
        )

        p_zone_data = zone_level[priority_zone]

        # Extract real-time alerts
        alerts = []
        for tid, v in vehicle_level.items():
            preds = v["prediction"]["details"]
            if preds and v["prediction"]["prediction_ready"] and not v["prediction"]["is_uncertain"]:
                if preds.get("risk_level", {}).get("label") == "HIGH":
                    alerts.append({
                        "alert_id": f"HYBRID-ALERT-T{tid}-F{frame_num}",
                        "track_id": tid,
                        "type": "HIGH_RISK_TRAJECTORY",
                        "severity": "CRITICAL" if preds.get("has_infraction", {}).get("violation_detected") else "HIGH",
                        "zone_id": v["zone_id"],
                        "message": f"Vehicle #{tid} flagged with high risk in {v['zone_id']}: {v['prediction']['safety_advisory']}",
                        "timestamp": time_seconds,
                    })
                elif preds.get("has_infraction", {}).get("violation_detected"):
                    itype = preds.get("infraction_type", {}).get("label", "INFRACTION")
                    alerts.append({
                        "alert_id": f"HYBRID-INFRACTION-T{tid}-F{frame_num}",
                        "track_id": tid,
                        "type": f"PREDICTED_{itype}",
                        "severity": "HIGH",
                        "zone_id": v["zone_id"],
                        "message": f"Vehicle #{tid} predicted {itype} infraction with confidence {preds.get('has_infraction', {}).get('confidence', 0.0)*100:.1f}%.",
                        "timestamp": time_seconds,
                    })

        system_level = {
            "frame": frame_num,
            "timestamp": time_seconds,
            "total_active_vehicles": total_active,
            "ready_predictions_count": ready_total,
            "warming_up_count": warmup_total,
            "system_fused_congestion_score": sys_fused_score,
            "priority_zone": priority_zone,
            "priority_zone_details": {
                "zone_id": priority_zone,
                "description": p_zone_data["description"],
                "fused_congestion_score": p_zone_data["derived_intelligence"]["fused_congestion_score"],
                "fused_level": p_zone_data["derived_intelligence"]["fused_congestion_level"],
                "deterministic_trend": p_zone_data["measured_values"]["deterministic_trend"],
                "recommended_action": p_zone_data["derived_intelligence"]["recommended_action"],
                "safety_status": p_zone_data["derived_intelligence"]["safety_status"],
            },
            "active_hybrid_alerts_count": len(alerts),
            "active_hybrid_alerts": alerts[:10],  # Top alerts
        }

        return {
            "frame": frame_num,
            "time_seconds": time_seconds,
            "system_level": system_level,
            "zone_level": zone_level,
            "lane_level": lane_level,
            "vehicle_level": vehicle_level,
        }
