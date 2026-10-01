"""
Traffic Intelligence Engine
===========================
Core analytical layer for the Smart Traffic Management System (Authority Mode).
Preserves existing CV datasets, zone geometry, rolling statistics, trend analysis,
and deterministic heuristic decision rules.
"""

import os
import json
from collections import deque
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
import pandas as pd

from ml.hybrid_traffic.realtime_inference import RealTimeTrafficPredictor
from ml.hybrid_traffic.traffic_aggregator import TrafficIntelligenceAggregator


class TrafficIntelligenceEngine:
    def __init__(self, workspace_dir="."):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.video_file = os.path.join(self.workspace_dir, "traffic2.mp4")
        self.tracks_file = os.path.join(self.workspace_dir, "traffic_tracks.csv")
        self.fps = 30.0
        self.total_frames = 1530
        self.history_seconds = 10
        self.trend_seconds = 3
        self.history_frames = int(self.history_seconds * self.fps)  # 300 frames
        self.trend_frames = int(self.trend_seconds * self.fps)      # 90 frames

        # Vehicle class labels
        self.class_names = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        # Zone coordinate definition files
        self.zone_files = {
            "ZONE 1": "lane_zone_points.npy",
            "ZONE 2": "zone_2_points.npy",
            "ZONE 3": "zone_3_points.npy",
            "ZONE 4": "zone_4_points.npy",
            "ZONE 5": "zone_5_points.npy",
            "ZONE 6": "zone_6_points.npy"
        }

        self.zone_descriptions = {
            "ZONE 1": "Main Approach Corridor",
            "ZONE 2": "Eastbound Exit / Turning Lane",
            "ZONE 3": "Intersection Core Junction",
            "ZONE 4": "Southbound Queue Area",
            "ZONE 5": "Westbound Inflow Lane",
            "ZONE 6": "Southeast Exit Lane"
        }

        # State storage
        self.zones = {}
        self.zone_polygons_list = {}
        self.frames_data = {}
        self.precomputed_telemetry = {}
        self.global_analytics = {}

        # Hybrid Model Integration (Phase 6)
        self.hybrid_predictor = RealTimeTrafficPredictor(
            checkpoint_path=os.path.join(self.workspace_dir, "checkpoints", "best_model.pth"),
            scaler_path=os.path.join(self.workspace_dir, "checkpoints", "feature_scaler.joblib"),
            config_path=os.path.join(self.workspace_dir, "checkpoints", "best_model_config.json"),
            device="cpu"
        )
        self.hybrid_aggregator = TrafficIntelligenceAggregator()

        # Initialize
        self._load_zones()
        self._load_tracking_data()
        self._precompute_all_frames()
        self._load_supplementary_analytics()

    def _load_zones(self):
        """Loads all zone polygons from .npy coordinate files."""
        for zone_name, filename in self.zone_files.items():
            path = os.path.join(self.workspace_dir, filename)
            if os.path.exists(path):
                points = np.load(path).astype(np.int32)
                self.zones[zone_name] = points
                self.zone_polygons_list[zone_name] = points.tolist()
            else:
                print(f"[TrafficIntelligence] Warning: Zone file missing: {filename}")

        print(f"[TrafficIntelligence] Loaded {len(self.zones)} active zones.")

    def _load_tracking_data(self):
        """Loads and pre-indexes all YOLOv8 + ByteTrack detections by frame number."""
        if not os.path.exists(self.tracks_file):
            print(f"[TrafficIntelligence] Error: {self.tracks_file} not found.")
            return

        df = pd.read_csv(self.tracks_file)
        df["frame"] = df["frame"].astype(int)

        # Precompute contact points and class names
        df["vehicle_type"] = df["class"].map(self.class_names).fillna("vehicle")
        df["bottom_center_x"] = ((df["x1"] + df["x2"]) / 2.0).round(1)
        df["bottom_center_y"] = df["y2"].round(1)

        # Assign zone containment to every detection
        detections_by_frame = {}
        grouped = df.groupby("frame")

        for frame_num, group in grouped:
            frame_records = []
            for _, row in group.iterrows():
                cx = float(row["bottom_center_x"])
                cy = float(row["bottom_center_y"])
                assigned_zone = None

                # Test containment against zones
                for z_name, polygon in self.zones.items():
                    if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                        assigned_zone = z_name
                        break

                frame_records.append({
                    "track_id": int(row["track_id"]),
                    "class_id": int(row["class"]),
                    "vehicle_type": row["vehicle_type"],
                    "x1": round(float(row["x1"]), 1),
                    "y1": round(float(row["y1"]), 1),
                    "x2": round(float(row["x2"]), 1),
                    "y2": round(float(row["y2"]), 1),
                    "center_x": round((float(row["x1"]) + float(row["x2"])) / 2.0, 1),
                    "center_y": round((float(row["y1"]) + float(row["y2"])) / 2.0, 1),
                    "bottom_x": cx,
                    "bottom_y": cy,
                    "zone": assigned_zone
                })
            detections_by_frame[int(frame_num)] = frame_records

        self.frames_data = detections_by_frame
        print(f"[TrafficIntelligence] Pre-indexed {len(df)} tracking points across {len(detections_by_frame)} frames.")

    def _classify_level(self, count):
        """Standard project thresholds: <= 2 LOW, 3-5 MEDIUM, >= 6 HIGH."""
        if count <= 2:
            return "LOW"
        elif count <= 5:
            return "MEDIUM"
        return "HIGH"

    def _get_level_color(self, level):
        """Returns standard UI hex color for each traffic level."""
        if level == "LOW":
            return "#10b981"    # Green
        elif level == "MEDIUM":
            return "#f59e0b"    # Amber
        return "#ef4444"        # Red

    def _calculate_trend(self, history_list):
        """
        Calculates trend over trend_frames (3 seconds = 90 frames).
        Compares recent 90-frame mean vs previous 90-frame mean with a 0.75 vehicle tolerance.
        """
        if len(history_list) < self.trend_frames * 2:
            return "STABLE"

        recent = np.mean(history_list[-self.trend_frames:])
        previous = np.mean(history_list[-self.trend_frames * 2: -self.trend_frames])
        difference = recent - previous

        if difference > 0.75:
            return "INCREASING"
        elif difference < -0.75:
            return "DECREASING"
        return "STABLE"

    def _get_recommendation(self, level, trend):
        """Decision recommendation matrix."""
        if level == "HIGH":
            if trend == "INCREASING":
                return "CONGESTION BUILDING"
            elif trend == "DECREASING":
                return "CONGESTION CLEARING"
            return "SUSTAINED HIGH TRAFFIC"
        elif level == "MEDIUM":
            if trend == "INCREASING":
                return "MONITOR - BUILDING"
            elif trend == "DECREASING":
                return "MONITOR - CLEARING"
            return "MONITOR"
        return "NORMAL FLOW"

    def _precompute_all_frames(self):
        """
        Precomputes continuous rolling state and hybrid intelligence across all 1530 frames.
        Caches to disk for instant sub-second server startup on subsequent executions.
        """
        cache_path = os.path.join(self.workspace_dir, "data", "telemetry_cache.pkl")
        if os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    self.precomputed_telemetry = pickle.load(f)
                print(f"[TrafficIntelligence] Loaded {len(self.precomputed_telemetry)} frames from telemetry cache.")
                return
            except Exception as e:
                print(f"[TrafficIntelligence] Cache load failed ({e}), recomputing...")

        zone_histories = {z: deque(maxlen=self.history_frames) for z in self.zones}
        zone_peaks = {z: 0 for z in self.zones}
        zone_high_starts = {z: None for z in self.zones}

        level_score_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        trend_score_map = {"DECREASING": 0, "STABLE": 1, "INCREASING": 2}

        for frame_num in range(self.total_frames):
            detections = self.frames_data.get(frame_num, [])

            # Count vehicles currently inside each zone
            zone_current_counts = {z: 0 for z in self.zones}
            for det in detections:
                if det["zone"] in zone_current_counts:
                    zone_current_counts[det["zone"]] += 1

            zone_telemetry = {}
            for z_name in self.zones:
                current_c = zone_current_counts[z_name]
                zone_histories[z_name].append(current_c)
                history_vals = list(zone_histories[z_name])

                avg_count = float(np.mean(history_vals)) if history_vals else 0.0
                zone_peaks[z_name] = max(zone_peaks[z_name], current_c)
                level = self._classify_level(round(avg_count))
                trend = self._calculate_trend(history_vals)

                # Sustained duration tracking
                if level == "HIGH":
                    if zone_high_starts[z_name] is None:
                        zone_high_starts[z_name] = frame_num / self.fps
                    sustained_sec = round((frame_num / self.fps) - zone_high_starts[z_name], 1)
                else:
                    zone_high_starts[z_name] = None
                    sustained_sec = 0.0

                recommendation = self._get_recommendation(level, trend)

                zone_telemetry[z_name] = {
                    "zone_name": z_name,
                    "description": self.zone_descriptions.get(z_name, ""),
                    "polygon": self.zone_polygons_list[z_name],
                    "current_count": current_c,
                    "average_count": round(avg_count, 1),
                    "peak_count": zone_peaks[z_name],
                    "level": level,
                    "trend": trend,
                    "sustained_seconds": sustained_sec,
                    "recommendation": recommendation,
                    "color": self._get_level_color(level)
                }

            # Deterministic Priority Zone Election
            priority_zone = max(
                zone_telemetry.keys(),
                key=lambda z: (
                    level_score_map[zone_telemetry[z]["level"]],
                    trend_score_map[zone_telemetry[z]["trend"]],
                    zone_telemetry[z]["average_count"],
                    zone_telemetry[z]["sustained_seconds"]
                )
            )

            p_info = zone_telemetry[priority_zone]

            # Determine actionable reason for priority
            if p_info["level"] == "HIGH" and p_info["trend"] == "INCREASING":
                reason = "Very high vehicle density with continuous influx detected."
                decision_action = "Increase Green Phase Duration (+15s Extension)"
            elif p_info["level"] == "HIGH":
                reason = "Sustained high volume requiring queue clearance."
                decision_action = "Maintain Extended Green Split"
            elif p_info["level"] == "MEDIUM" and p_info["trend"] == "INCREASING":
                reason = "Traffic volume accumulating above baseline."
                decision_action = "Prepare Adaptive Green Extension"
            else:
                reason = "Nominal flow conditions across primary corridors."
                decision_action = "Standard Cycle Split"

            priority_details = {
                "zone": priority_zone,
                "description": p_info["description"],
                "level": p_info["level"],
                "trend": p_info["trend"],
                "average": p_info["average_count"],
                "peak": p_info["peak_count"],
                "recommendation": p_info["recommendation"],
                "reason": reason,
                "action": decision_action,
                "color": p_info["color"]
            }

            # --- Phase 6 Real-Time Hybrid Model Inference & Hierarchical Aggregation ---
            hybrid_preds = self.hybrid_predictor.predict_frame_tracks(frame_num, detections)
            hybrid_intel = self.hybrid_aggregator.aggregate(
                frame_num=frame_num,
                time_seconds=round(frame_num / self.fps, 2),
                measured_detections=detections,
                model_predictions=hybrid_preds,
                deterministic_zones=zone_telemetry,
            )

            # Enrich detections with vehicle-level hybrid predictions
            vehicle_preds = hybrid_intel.get("vehicle_level", {})
            enriched_detections = []
            for det in detections:
                det_copy = dict(det)
                tid = det_copy["track_id"]
                if tid in vehicle_preds:
                    v_meta = vehicle_preds[tid]
                    det_copy["prediction_ready"] = v_meta["prediction"]["prediction_ready"]
                    det_copy["status"] = v_meta["prediction"]["status"]
                    det_copy["warmup_progress"] = v_meta["prediction"]["warmup_progress"]
                    det_copy["confidence"] = v_meta["prediction"]["confidence"]
                    det_copy["is_uncertain"] = v_meta["prediction"]["is_uncertain"]
                    det_copy["safety_advisory"] = v_meta["prediction"]["safety_advisory"]
                    det_copy["hybrid_predictions"] = v_meta["prediction"]["details"]
                else:
                    det_copy["prediction_ready"] = False
                    det_copy["status"] = "WARMING_UP"
                    det_copy["warmup_progress"] = 0.0
                    det_copy["confidence"] = 0.0
                    det_copy["is_uncertain"] = True
                    det_copy["safety_advisory"] = "Collecting history"
                    det_copy["hybrid_predictions"] = None
                enriched_detections.append(det_copy)

            # Enhance priority details with fused metrics
            sys_priority = hybrid_intel["system_level"]["priority_zone_details"]
            priority_details["fused_score"] = sys_priority["fused_congestion_score"]
            priority_details["fused_level"] = sys_priority["fused_level"]
            priority_details["safety_status"] = sys_priority["safety_status"]

            # Overall System Summary
            total_active = len(detections)
            busiest_zone = max(zone_telemetry.keys(), key=lambda z: zone_telemetry[z]["average_count"])

            self.precomputed_telemetry[frame_num] = {
                "frame": frame_num,
                "time_seconds": round(frame_num / self.fps, 2),
                "total_frames": self.total_frames,
                "fps": self.fps,
                "active_vehicles_count": total_active,
                "zones": zone_telemetry,
                "priority_zone": priority_zone,
                "priority_details": priority_details,
                "system_summary": {
                    "total_monitored_zones": len(self.zones),
                    "total_unique_vehicles": 697,  # from traffic_smoothed_summary.csv
                    "active_in_frame": total_active,
                    "highest_traffic_zone": busiest_zone,
                    "highest_traffic_level": zone_telemetry[busiest_zone]["level"],
                    "intersection_trend": p_info["trend"],
                    "current_priority": priority_zone,
                    "fused_system_congestion": hybrid_intel["system_level"]["system_fused_congestion_score"],
                    "active_hybrid_alerts_count": hybrid_intel["system_level"]["active_hybrid_alerts_count"],
                },
                "hybrid_intelligence": hybrid_intel,
                "detections": enriched_detections
            }

        try:
            import pickle
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.precomputed_telemetry, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[TrafficIntelligence] Telemetry cache persisted to {cache_path}.")
        except Exception as e:
            print(f"[TrafficIntelligence] Warning: Failed to save cache: {e}")

        print("[TrafficIntelligence] Precomputation complete. All 1530 frames ready for O(1) streaming with Hybrid Intelligence.")

    def _load_supplementary_analytics(self):
        """Loads real offline analytical data from CSV and JSON files for dashboard charts."""
        analytics = {
            "vehicle_modal_distribution": {
                "cars": 160,
                "motorcycles": 299,
                "buses": 31,
                "trucks": 56
            },
            "timeline_flow": [],
            "congestion_metrics": {},
            "policy_report": []
        }

        # 1. Flow data
        flow_path = os.path.join(self.workspace_dir, "zone_traffic_flow.csv")
        if os.path.exists(flow_path):
            flow_df = pd.read_csv(flow_path)
            analytics["timeline_flow"] = [
                {
                    "interval": f"{int(row['10_second_interval']) * 10}-{(int(row['10_second_interval']) + 1) * 10}s",
                    "vehicles": int(row["unique_vehicles"])
                }
                for _, row in flow_df.iterrows()
            ]

        # 2. Congestion analysis
        cg_path = os.path.join(self.workspace_dir, "congestion_analysis.csv")
        if os.path.exists(cg_path):
            cg_df = pd.read_csv(cg_path)
            if not cg_df.empty:
                r = cg_df.iloc[0]
                analytics["congestion_metrics"] = {
                    "density_score": float(r["density_score"]),
                    "flow_score": float(r["flow_score"]),
                    "variation_score": float(r["variation_score"]),
                    "congestion_score": float(r["congestion_score"]),
                    "congestion_level": str(r["congestion_level"]),
                    "peak_vehicles": int(r["peak_interval_vehicles"])
                }

        # 3. Policy decisions
        pol_path = os.path.join(self.workspace_dir, "traffic_policy_report.csv")
        if os.path.exists(pol_path):
            pol_df = pd.read_csv(pol_path)
            analytics["policy_report"] = pol_df.to_dict(orient="records")

        self.global_analytics = analytics

    def get_frame_data(self, frame_num):
        """Returns instantaneous telemetry and detections for a specific frame."""
        frame_idx = max(0, min(int(frame_num), self.total_frames - 1))
        return self.precomputed_telemetry.get(frame_idx, {})

    def get_zones_metadata(self):
        """Returns static metadata and polygon definitions for all zones."""
        return {
            z_name: {
                "name": z_name,
                "description": self.zone_descriptions.get(z_name, ""),
                "polygon": self.zone_polygons_list[z_name]
            }
            for z_name in self.zones
        }

    def get_global_analytics(self):
        """Returns real aggregated analytical datasets for frontend charts."""
        # Calculate zone aggregate statistics across the full video
        zone_aggregates = []
        for z_name in sorted(self.zones.keys()):
            counts = [
                self.precomputed_telemetry[f]["zones"][z_name]["current_count"]
                for f in range(self.total_frames)
            ]
            zone_aggregates.append({
                "zone": z_name,
                "description": self.zone_descriptions.get(z_name, ""),
                "mean_count": round(float(np.mean(counts)), 2),
                "peak_count": int(np.max(counts)),
                "p95_count": round(float(np.percentile(counts, 95)), 1)
            })

        return {
            "zone_aggregates": zone_aggregates,
            "modal_distribution": self.global_analytics.get("vehicle_modal_distribution", {}),
            "timeline_flow": self.global_analytics.get("timeline_flow", []),
            "congestion_metrics": self.global_analytics.get("congestion_metrics", {}),
            "policy_report": self.global_analytics.get("policy_report", [])
        }

    def predict_live_track(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Runs live inference on a single vehicle observation."""
        return self.hybrid_predictor.predict_live(observation)

    def predict_live_frame(self, frame_num: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes an arbitrary live video frame through the hybrid pipeline."""
        preds = self.hybrid_predictor.predict_frame_tracks(frame_num, detections)
        return self.hybrid_aggregator.aggregate(
            frame_num=frame_num,
            time_seconds=frame_num / self.fps,
            measured_detections=detections,
            model_predictions=preds,
            deterministic_zones=None,
        )

    def get_hybrid_status(self) -> Dict[str, Any]:
        """Returns hybrid model operational telemetry and status."""
        return {
            "status": "ACTIVE",
            "model_architecture": "TCNTransformerHybrid",
            "checkpoint": self.hybrid_predictor.checkpoint_path,
            "scaler": self.hybrid_predictor.scaler_path,
            "epoch": self.hybrid_predictor.epoch,
            "val_loss": self.hybrid_predictor.val_loss,
            "sequence_length": self.hybrid_predictor.sequence_length,
            "feature_count": self.hybrid_predictor.feature_count,
            "telemetry": self.hybrid_predictor.get_telemetry(),
        }
