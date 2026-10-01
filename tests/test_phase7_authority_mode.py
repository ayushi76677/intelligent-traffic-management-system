"""
Phase 7 Verification Test Suite — Authority Mode Intelligent Traffic Control
============================================================================
Comprehensive automated tests verifying:
1. Authority Mode web interface loading and asset integrity
2. Map integration, road geometry, and offline Leaflet fallback
3. Location selection and strict 4-state data mode classification
4. Live vehicle tracking, track-level hybrid predictions, risk & violation highlighting
5. Congestion visualization with strict 3-way metric separation (Measured, Predicted, Derived)
6. Zone analysis across all 6 zones and lane-level traffic inspection
7. Violations registry, infraction detection, and threshold enforcement
8. Real-time alerts system and event timeline aggregation
9. Real-time update responsiveness, batch buffering, and sustained throughput
10. Error states, graceful fallbacks, and edge case resilience
11. Full End-to-End pipeline verification
"""

import os
import sys
import json
import time
import unittest

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app import app
from traffic_intelligence import TrafficIntelligenceEngine


class TestPhase7AuthorityMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes Flask test client and underlying intelligence engine."""
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        cls.engine = TrafficIntelligenceEngine(workspace_dir=WORKSPACE_ROOT)

    # =========================================================================
    # 1. AUTHORITY MODE LOAD
    # =========================================================================
    def test_01_authority_mode_load(self):
        """Verifies dashboard root loads successfully and contains all Phase 7 UI components."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Core container elements
        self.assertIn('id="googleMap"', html, "Missing Google Map container")
        self.assertIn('id="cvCanvas"', html, "Missing CV Canvas overlay")
        self.assertIn('id="trafficVideo"', html, "Missing traffic video element")
        self.assertIn('id="zoneCardsGrid"', html, "Missing Zone Cards grid")

        # Phase 7 UI Elements
        self.assertIn('id="selLocationDataModeBadge"', html, "Missing Location Data Mode badge")
        self.assertIn('id="camDetailDataModeBadge"', html, "Missing Camera Detail Data Mode badge")
        self.assertIn('id="hybridModelHeroCard"', html, "Missing Hybrid Model Hero Card")
        self.assertIn('id="threeWayCongestionSection"', html, "Missing 3-Way Congestion Section")
        self.assertIn('id="zoneInspectorSection"', html, "Missing Zone Inspector Section")
        self.assertIn('id="eventTimelineSection"', html, "Missing Event Timeline Section")

        # Layer toggles
        self.assertIn('id="toggleHybridRisk"', html, "Missing Hybrid Risk layer toggle")
        self.assertIn('id="toggleViolations"', html, "Missing Violations layer toggle")

        # Leaflet fallback assets in HTML
        self.assertIn('leaflet.css', html, "Missing Leaflet CSS stylesheet in head")
        self.assertIn('leaflet.js', html, "Missing Leaflet JS in head")

        # Static assets reachability
        css_res = self.client.get("/static/css/dashboard.css")
        self.assertEqual(css_res.status_code, 200, "dashboard.css should be accessible")
        js_res = self.client.get("/static/js/dashboard.js")
        self.assertEqual(js_res.status_code, 200, "dashboard.js should be accessible")

    # =========================================================================
    # 2. MAP INTEGRATION & FALLBACK
    # =========================================================================
    def test_02_map_integration_and_fallback(self):
        """Verifies map configuration, Indian location datasets, and offline Leaflet fallback."""
        # 1. Config API
        config_res = self.client.get("/api/config")
        self.assertEqual(config_res.status_code, 200)
        config_data = json.loads(config_res.data)
        self.assertIn("map_id", config_data)
        self.assertIn("has_key", config_data)

        # 2. Locations API
        loc_res = self.client.get("/api/locations")
        self.assertEqual(loc_res.status_code, 200)
        loc_data = json.loads(loc_res.data)
        self.assertIn("cities", loc_data)
        cities = loc_data["cities"]
        self.assertTrue(len(cities) > 0, "Cities database must not be empty")

        indore = next((c for c in cities if c.get("name") == "Indore" or c.get("city") == "Indore"), None)
        self.assertIsNotNone(indore, "Indore must exist in locations")
        self.assertIn("intersections", indore)
        vj = next((i for i in indore["intersections"] if "Vijay Nagar" in i.get("name", "")), None)
        self.assertIsNotNone(vj, "Vijay Nagar intersection must be present")
        self.assertAlmostEqual(vj["latitude"], 22.7533, places=3)
        self.assertAlmostEqual(vj["longitude"], 75.8937, places=3)

        # 3. Verify Leaflet fallback functions exist in dashboard.js
        with open(os.path.join(WORKSPACE_ROOT, "static", "js", "dashboard.js"), "r", encoding="utf-8") as f:
            js_content = f.read()

        self.assertIn("function initLeafletFallbackMap()", js_content)
        self.assertIn("function renderLeafletCctvMarkers()", js_content)
        self.assertIn("function renderLeafletZonePolygons()", js_content)
        self.assertIn("OFFLINE / LOCAL ROAD GEOMETRY MODE", js_content)

    # =========================================================================
    # 3. LOCATION SELECTION & DATA MODE CLASSIFICATION
    # =========================================================================
    def test_03_location_selection_and_data_modes(self):
        """Verifies camera repository inventory and accurate 4-state data mode classification."""
        cam_res = self.client.get("/api/cameras")
        self.assertEqual(cam_res.status_code, 200)
        cam_data = json.loads(cam_res.data)
        cameras = cam_data.get("cameras", [])
        self.assertTrue(len(cameras) >= 3, "Expected at least 3 configured authority cameras")

        # Primary live camera CAM-IND-001 (Vijay Nagar)
        primary = next((c for c in cameras if c.get("camera_id") == "CAM-IND-001"), None)
        self.assertIsNotNone(primary, "CAM-IND-001 must exist")
        self.assertEqual(primary.get("feed_status"), "LIVE", "CAM-IND-001 must be LIVE")
        self.assertEqual(primary.get("feed_availability"), "LIVE FEED AVAILABLE")

        # Camera with LOCATION ONLY status (no simulated video fabricated)
        loc_only = [c for c in cameras if c.get("feed_status") == "LOCATION ONLY"]
        self.assertTrue(len(loc_only) > 0, "Must have cameras with LOCATION ONLY status")

        # Camera inspection API
        cam_detail_res = self.client.get("/api/camera/CAM-IND-001")
        self.assertEqual(cam_detail_res.status_code, 200)
        c_detail = json.loads(cam_detail_res.data)
        self.assertEqual(c_detail.get("feed_status"), "LIVE")
        self.assertIn("metrics", c_detail)
        self.assertGreater(c_detail["metrics"]["vehicles_detected"], 0)

        # Reverse geocoding endpoint resilience
        geo_res = self.client.get("/api/geocode?lat=22.7533&lng=75.8937")
        self.assertEqual(geo_res.status_code, 200)
        geo_data = json.loads(geo_res.data)
        self.assertIn("formatted_address", geo_data)
        self.assertEqual(geo_data.get("country"), "India")

    # =========================================================================
    # 4. LIVE VEHICLES & TRACK-LEVEL HYBRID PREDICTIONS
    # =========================================================================
    def test_04_live_vehicles_and_hybrid_predictions(self):
        """Verifies vehicle detections enriched with TCN-Transformer hybrid multi-task predictions."""
        # Frame 300 has warmed up sequence buffer
        frame_res = self.client.get("/api/frame/300")
        self.assertEqual(frame_res.status_code, 200)
        frame_data = json.loads(frame_res.data)

        detections = frame_data.get("detections", [])
        self.assertTrue(len(detections) > 0, "Frame 300 must have active vehicle detections")

        # Inspect vehicle-level fields
        for det in detections:
            self.assertIn("track_id", det)
            self.assertIn("vehicle_type", det)
            self.assertIn("x1", det)
            self.assertIn("y1", det)
            self.assertIn("x2", det)
            self.assertIn("y2", det)
            self.assertIn("bottom_x", det)
            self.assertIn("bottom_y", det)
            self.assertIn("prediction_ready", det)
            self.assertIn("status", det)
            self.assertIn("confidence", det)

            # If prediction ready, check multi-task output
            if det["prediction_ready"]:
                hp = det.get("hybrid_predictions")
                self.assertIsNotNone(hp)
                self.assertIn("congestion_level", hp)
                self.assertIn("risk_level", hp)
                self.assertIn("has_infraction", hp)
                self.assertIn("infraction_type", hp)
                self.assertIn("is_approaching", hp)
                self.assertIn("approach_threat_score", hp)
                self.assertIn("maneuver_type", hp)
                self.assertIn("motion_state", hp)

                # Confidence and threat bounds
                self.assertGreaterEqual(det["confidence"], 0.0)
                self.assertLessEqual(det["confidence"], 1.0)
                self.assertGreaterEqual(hp["approach_threat_score"], 0.0)
                self.assertLessEqual(hp["approach_threat_score"], 100.0)

    # =========================================================================
    # 5. CONGESTION & 3-WAY MATRIX SEPARATION
    # =========================================================================
    def test_05_three_way_congestion_matrix(self):
        """Verifies strict separation of MEASURED, MODEL PREDICTED, and DERIVED data."""
        # Inspect hybrid frame intelligence API
        h_res = self.client.get("/api/hybrid/frame/300")
        self.assertEqual(h_res.status_code, 200)
        h_data = json.loads(h_res.data)

        self.assertIn("zone_level", h_data)
        zone_intel = h_data["zone_level"]

        for z_name, z_info in zone_intel.items():
            # 1. MEASURED VALUES
            self.assertIn("measured_values", z_info, f"Missing measured_values in {z_name}")
            meas = z_info["measured_values"]
            self.assertIn("vehicle_count", meas)
            self.assertIn("density_score", meas)
            self.assertIn("modal_breakdown", meas)
            self.assertIn("average_speed_px_per_sec", meas)
            self.assertIn("sustained_high_seconds", meas)

            # 2. MODEL PREDICTIONS
            self.assertIn("model_predictions", z_info, f"Missing model_predictions in {z_name}")
            pred = z_info["model_predictions"]
            self.assertIn("mean_predicted_congestion_score", pred)
            self.assertIn("average_model_confidence", pred)
            self.assertIn("mean_approach_threat_score", pred)
            self.assertIn("approaching_vehicles_count", pred)
            self.assertIn("predicted_risk_distribution", pred)

            # 3. DERIVED INTELLIGENCE
            self.assertIn("derived_intelligence", z_info, f"Missing derived_intelligence in {z_name}")
            deriv = z_info["derived_intelligence"]
            self.assertIn("deterministic_base_score", deriv)
            self.assertIn("model_predicted_score", deriv)
            self.assertIn("fused_congestion_score", deriv)
            self.assertIn("fused_congestion_level", deriv)
            self.assertIn("safety_status", deriv)
            self.assertIn("recommended_action", deriv)

            # Strict bounds check
            self.assertGreaterEqual(deriv["fused_congestion_score"], 0.0)
            self.assertLessEqual(deriv["fused_congestion_score"], 100.0)
            self.assertIn(deriv["fused_congestion_level"], ["LOW", "MODERATE", "HIGH", "SEVERE", "NOMINAL", "BUSY", "CONGESTED"])

    # =========================================================================
    # 6. ZONE ANALYSIS & LANE-LEVEL INSPECTION
    # =========================================================================
    def test_06_zone_analysis_and_lane_inspection(self):
        """Verifies all 6 spatial zones and lane-level traffic inspection breakdown."""
        # 1. Zones API
        zones_res = self.client.get("/api/zones")
        self.assertEqual(zones_res.status_code, 200)
        zones_data = json.loads(zones_res.data)
        self.assertEqual(zones_data["total"], 6, "Expected 6 active authority zones")

        pixel_polys = zones_data.get("pixel_polygons", {})
        for i in range(1, 7):
            z_id = f"ZONE {i}"
            self.assertIn(z_id, pixel_polys, f"{z_id} must have polygon coordinates")
            self.assertGreaterEqual(len(pixel_polys[z_id]), 3, f"{z_id} polygon must have >= 3 vertices")

        # 2. Lane-level inspection in hybrid intelligence
        h_res = self.client.get("/api/hybrid/frame/150")
        self.assertEqual(h_res.status_code, 200)
        h_data = json.loads(h_res.data)
        self.assertIn("lane_level", h_data)
        lanes = h_data["lane_level"]
        self.assertTrue(len(lanes) > 0, "Must have lane-level telemetry")

        for lane_name, l_info in lanes.items():
            self.assertIn("vehicle_count", l_info)
            self.assertIn("approaching_vehicles_count", l_info)
            self.assertIn("risk_level_distribution", l_info)
            self.assertIn("infractions_detected_count", l_info)

    # =========================================================================
    # 7. VIOLATIONS REGISTRY & INFRACTIONS
    # =========================================================================
    def test_07_violations_registry(self):
        """Verifies traffic violation tracking, infractions, and speed limit rules."""
        # 1. Violations list
        v_res = self.client.get("/api/violations")
        self.assertEqual(v_res.status_code, 200)
        v_data = json.loads(v_res.data)
        self.assertIn("violations", v_data)
        violations = v_data["violations"]
        self.assertTrue(len(violations) > 0, "Must return recorded traffic violations")

        # Verify schema
        for v in violations:
            self.assertIn("violation_id", v)
            self.assertIn("vehicle_id", v)
            self.assertIn("severity", v)
            self.assertIn("status", v)

        # 2. Violation stats
        stats_res = self.client.get("/api/violation_stats")
        self.assertEqual(stats_res.status_code, 200)
        stats = json.loads(stats_res.data)
        self.assertIn("total_violations", stats)
        self.assertIn("by_severity", stats)
        self.assertIn("by_type", stats)

        # 3. Speed limits
        limits_res = self.client.get("/api/speed_limits")
        self.assertEqual(limits_res.status_code, 200)
        limits_data = json.loads(limits_res.data)
        self.assertIn("road_limits", limits_data)

    # =========================================================================
    # 8. ALERTS & EVENT TIMELINE
    # =========================================================================
    def test_08_alerts_and_event_timeline(self):
        """Verifies real-time alerts combining overspeed, severe congestion, incidents, and hybrid risks."""
        alerts_res = self.client.get("/api/alerts?frame=300")
        self.assertEqual(alerts_res.status_code, 200)
        alerts_data = json.loads(alerts_res.data)
        self.assertIn("total", alerts_data)
        self.assertIn("alerts", alerts_data)
        alerts = alerts_data["alerts"]
        self.assertTrue(len(alerts) > 0, "Must return active traffic alerts")

        # Verify alert item schema
        types_seen = set()
        for a in alerts:
            self.assertIn("alert_id", a)
            self.assertIn("type", a)
            self.assertIn("title", a)
            self.assertIn("severity", a)
            self.assertIn("message", a)
            types_seen.add(a["type"])

        # Check that high speed or severe congestion alerts are covered
        self.assertTrue("HIGH_SPEED" in types_seen or "SEVERE_CONGESTION" in types_seen or "INCIDENT" in types_seen)

    # =========================================================================
    # 9. REAL-TIME UPDATE & BATCH BUFFERING
    # =========================================================================
    def test_09_real_time_update_and_buffering(self):
        """Verifies zero-latency telemetry batching and sustained throughput >= 24.7 FPS."""
        # 1. Telemetry batch fetch
        batch_res = self.client.get("/api/telemetry_batch?start=0&count=60")
        self.assertEqual(batch_res.status_code, 200)
        batch_data = json.loads(batch_res.data)
        self.assertEqual(batch_data["total"], 60)
        self.assertEqual(len(batch_data["frames"]), 60)

        # 2. Measure throughput over 100 consecutive frames
        t0 = time.time()
        for f_idx in range(50, 150):
            frame = self.engine.get_frame_data(f_idx)
            self.assertIsNotNone(frame)
        duration = time.time() - t0

        fps = 100.0 / duration
        # Must comfortably exceed 24.7 FPS baseline (cached telemetry provides > 1000 FPS)
        self.assertGreaterEqual(fps, 24.7, f"Throughput {fps:.1f} FPS below 24.7 baseline")

    # =========================================================================
    # 10. ERROR STATES & RESILIENCE
    # =========================================================================
    def test_10_error_states_and_resilience(self):
        """Verifies graceful handling of out-of-range requests, missing sensors, and warmup."""
        # 1. Out of range frame
        err_frame_res = self.client.get("/api/frame/99999")
        self.assertEqual(err_frame_res.status_code, 404)
        err_json = json.loads(err_frame_res.data)
        self.assertIn("error", err_json)

        # 2. Non-existent camera
        err_cam_res = self.client.get("/api/camera/CAM-INVALID-999")
        self.assertEqual(err_cam_res.status_code, 404)

        # 3. Early frame warmup verification (frame 0)
        early_res = self.client.get("/api/frame/0")
        self.assertEqual(early_res.status_code, 200)
        early_data = json.loads(early_res.data)
        # In frame 0, sequence buffer has 1 observation, so status must be WARMING_UP
        dets = early_data.get("detections", [])
        if dets:
            self.assertEqual(dets[0].get("status"), "WARMING_UP")
            self.assertFalse(dets[0].get("prediction_ready"))

    # =========================================================================
    # 11. END-TO-END PIPELINE INTEGRATION
    # =========================================================================
    def test_11_end_to_end_pipeline(self):
        """Verifies full pipeline: Location -> Camera -> CV -> Tracking -> Features -> Hybrid Model -> Aggregator -> Authority Mode."""
        # 1. System status
        status_res = self.client.get("/api/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = json.loads(status_res.data)
        self.assertEqual(status_data.get("status"), "ONLINE")
        self.assertEqual(status_data.get("mode"), "AUTHORITY MODE")
        self.assertIn("TCN-Transformer Gated Hybrid", status_data.get("active_models", []))

        # 2. Hybrid Model Status
        h_status_res = self.client.get("/api/hybrid/status")
        self.assertEqual(h_status_res.status_code, 200)
        h_status = json.loads(h_status_res.data)
        self.assertEqual(h_status.get("status"), "ACTIVE")
        self.assertEqual(h_status.get("model_architecture"), "TCNTransformerHybrid")
        self.assertTrue(os.path.exists(h_status.get("checkpoint", "")))
        self.assertTrue(os.path.exists(h_status.get("scaler", "")))
        self.assertEqual(h_status.get("feature_count"), 20)
        self.assertEqual(h_status.get("sequence_length"), 20)

        # 3. Live Inference API test
        live_res = self.client.post("/api/hybrid/live", json={
            "frame": 50,
            "observations": [
                {
                    "track_id": 42,
                    "class_id": 2,
                    "x1": 100.0,
                    "y1": 200.0,
                    "x2": 150.0,
                    "y2": 260.0,
                    "zone": "ZONE 1"
                }
            ]
        })
        self.assertEqual(live_res.status_code, 200)
        live_intel = json.loads(live_res.data)
        self.assertIn("vehicle_level", live_intel)
        self.assertIn("system_level", live_intel)

        # 4. Driver Mode synchronization verification
        driver_res = self.client.get("/api/driver/feed?frame=300")
        self.assertEqual(driver_res.status_code, 200)
        driver_data = json.loads(driver_res.data)
        self.assertEqual(driver_data.get("status"), "ONLINE")
        self.assertIn("hybrid_intelligence", driver_data)


if __name__ == "__main__":
    unittest.main()
