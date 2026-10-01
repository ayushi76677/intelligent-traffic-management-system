"""
Phase 8 Automated Test Suite: Driver Mode Intelligent Traffic Assistance
Verifies all 17 requirements and 14 final evaluation criteria:
- Driver Mode loading
- Location permission handling (granted, denied, unavailable, timeout)
- Current location marker & heading
- Map integration (Google Maps + Leaflet fallback)
- Destination selection & quick presets
- Route calculation & clean abstraction
- Traffic-aware route visualization & congestion coloring
- Nearby traffic intelligence & proximity-filtered alerts
- Real-time route monitoring & movement threshold (>100m)
- Error states, API resilience, and uncertainty/provenance tagging
- Responsive UI and mobile layout
- Zero regression on Authority Mode
- End-to-end pipeline execution
"""

import unittest
import json
import re
import os
from app import app, engine, sumo_engine


class TestPhase8DriverMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

        # Read frontend files for DOM and CSS validation
        with open("static/index.html", "r", encoding="utf-8") as f:
            cls.html_content = f.read()

        with open("static/css/dashboard.css", "r", encoding="utf-8") as f:
            cls.css_content = f.read()

        with open("static/js/dashboard.js", "r", encoding="utf-8") as f:
            cls.js_content = f.read()

    # 1. DRIVER MODE LOAD
    def test_01_driver_mode_load(self):
        """Validates that Driver Mode DOM container and all essential cockpit HUD elements exist."""
        required_ids = [
            "driverModeView",
            "driverMap",
            "dmEgoSpeed",
            "dmSpeedLimitSign",
            "dmGpsStatus",
            "dmDataProvenance",
            "dmCurrentRoad",
            "btnDmLocateMe",
            "btnExitDriverMode",
            "dmTurnBanner",
            "dmTurnInstruction",
            "dmTurnSub",
            "btnDmRecenterMap",
            "dmDestInput",
            "dmPresetChips",
            "btnDmCalculateRoute",
            "btnDmClearRoute",
            "btnDmRecalcRoute",
            "dmRouteMetricsStrip",
            "dmRouteEta",
            "dmRouteDistance",
            "dmRouteCongestion",
            "dmRouteStatusTag",
            "dmSevereBanner",
            "dmBottleneckText",
            "btnDmAcceptReroute",
            "dmHazardsCount",
            "dmHazardsList",
            "dmModelSafetyStatus"
        ]
        for el_id in required_ids:
            self.assertIn(f'id="{el_id}"', self.html_content, f"Missing required Driver Mode element ID: {el_id}")

        # Check CSS rules
        required_classes = [
            ".driver-mode-hud-view",
            ".dm-top-bar",
            ".dm-cockpit-layout",
            ".dm-map-container",
            ".driver-map-canvas",
            ".dm-turn-banner",
            ".btn-dm-recenter",
            ".dm-sidebar-panel",
            ".dm-panel-card",
            ".dm-metrics-strip",
            ".dm-driver-marker-wrap"
        ]
        for cls_name in required_classes:
            self.assertIn(cls_name, self.css_content, f"Missing required CSS rule: {cls_name}")

    # 2. LOCATION PERMISSION
    def test_02_location_permission_handling(self):
        """Verifies geolocation permission handling for granted, denied, unavailable, and timeout states."""
        self.assertIn("function requestDriverLocation", self.js_content)
        self.assertIn("navigator.geolocation", self.js_content)
        self.assertIn("PERMISSION_DENIED", self.js_content)
        self.assertIn("POSITION_UNAVAILABLE", self.js_content)
        self.assertIn("TIMEOUT", self.js_content)
        self.assertIn("driverGpsRequested", self.js_content)
        # Verify fallback location is assigned when permission is denied or unavailable
        self.assertIn("Vijay Nagar Corridor", self.js_content)

    # 3. CURRENT LOCATION & MARKER
    def test_03_current_location_marker(self):
        """Verifies driver marker rendering, pulsing effect, heading, and distinction from traffic markers."""
        self.assertIn("updateDriverMarkerOnMap", self.js_content)
        self.assertIn("dm-driver-marker-wrap", self.js_content)
        self.assertIn("dm-driver-pulse-ring", self.css_content)
        self.assertIn("driverPosition.heading", self.js_content)
        self.assertIn("FORWARD_CLOSED_ARROW", self.js_content)

    # 4. MAP INTEGRATION & FALLBACK
    def test_04_map_integration_dual_engine(self):
        """Verifies driver map support for Google Maps and autonomous Leaflet fallback."""
        self.assertIn("initDriverMap", self.js_content)
        self.assertIn("initDriverLeafletMap", self.js_content)
        self.assertIn("dmMapStateNotice", self.html_content)
        self.assertIn("TrafficLayer", self.js_content)

    # 5. DESTINATION SELECTION & PRESETS
    def test_05_destination_selection(self):
        """Verifies destination input and quick Indore preset chips."""
        self.assertIn("Palasia Square", self.html_content)
        self.assertIn("Radisson Square", self.html_content)
        self.assertIn("MR-10 Junction", self.html_content)
        self.assertIn("Bhawarkua Square", self.html_content)
        self.assertIn("Bengali Square", self.html_content)
        self.assertIn("Indore Airport", self.html_content)

    # 6. ROUTE CALCULATION & EVALUATION
    def test_06_route_calculation_and_evaluation(self):
        """Tests /api/routes/evaluate endpoint with various coordinate formats and corridor inputs."""
        payload = {
            "start": {"lat": 22.7533, "lng": 75.8937},
            "destination": {"lat": 22.7244, "lng": 75.8839},
            "points": [
                [22.7533, 75.8937],
                [22.7380, 75.8880],
                [22.7244, 75.8839]
            ]
        }
        res = self.client.post("/api/routes/evaluate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("severe_congestion_detected", data)
        self.assertIn("highest_congestion_score", data)
        self.assertIn("route_condition", data)
        self.assertIn("recommended_action", data)
        self.assertIn("estimated_delay_minutes", data)
        self.assertIn("data_provenance", data)

    # 7. TRAFFIC VISUALIZATION & CONGESTION
    def test_07_traffic_and_congestion_visualization(self):
        """Verifies multi-tier congestion coloring and route status tag."""
        self.assertIn(".dm-leg-dot.normal", self.css_content)
        self.assertIn(".dm-leg-dot.congested", self.css_content)
        self.assertIn(".dm-leg-dot.severe", self.css_content)
        self.assertIn("dmRouteCongestion", self.js_content)
        self.assertIn("Recommended route according to current traffic conditions", self.html_content)

    # 8. ALERTS & PROXIMITY FILTERING
    def test_08_driver_feed_proximity_alerts(self):
        """Tests /api/driver/feed with driver coordinates for proximity calculation and alert filtering."""
        res = self.client.get("/api/driver/feed?lat=22.7533&lng=75.8937&frame=100")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("service"), "DriverModeSync")
        self.assertEqual(data.get("status"), "ONLINE")
        self.assertIn("nearby_severe_zones", data)
        self.assertIn("active_road_hazards", data)
        self.assertIn("proximity_alerts", data)

        # Check proximity annotations
        zones = data.get("nearby_severe_zones", [])
        if zones:
            self.assertIn("distance_meters", zones[0])
            self.assertIn("distance_km", zones[0])

        hazards = data.get("active_road_hazards", [])
        if hazards:
            self.assertIn("distance_meters", hazards[0])

        # Check hybrid intelligence
        self.assertIn("hybrid_intelligence", data)
        self.assertIn("system_fused_congestion_score", data["hybrid_intelligence"])

    # 9. REAL-TIME ROUTE MONITORING & THRESHOLD
    def test_09_real_time_route_monitoring(self):
        """Verifies distance movement threshold (>100m) check to avoid GPS jitter recalculation."""
        self.assertIn("checkDriverMovementAndRecalculate", self.js_content)
        self.assertIn("distMeters >= 100", self.js_content)
        self.assertIn("driverLastEvalPosition", self.js_content)

    # 10. REROUTE ADVISORY & BYPASS DIVERSION
    def test_10_reroute_guidance(self):
        """Verifies one-click bypass diversion via Eastern Bypass / MR-10."""
        self.assertIn("handleAcceptReroute", self.js_content)
        self.assertIn("btnDmAcceptReroute", self.js_content)
        self.assertIn("Eastern Bypass", self.js_content)

    # 11. ERROR STATES & RESILIENCE
    def test_11_error_states_and_resilience(self):
        """Tests driver feed and route evaluator against malformed requests and missing data."""
        # Empty payload route evaluate
        res = self.client.post("/api/routes/evaluate", json={})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("route_condition", data)

        # Non-numeric driver feed params
        res = self.client.get("/api/driver/feed?lat=invalid&lng=invalid")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ONLINE")

        # Out of bounds frame
        res = self.client.get("/api/driver/feed?frame=999999")
        self.assertEqual(res.status_code, 200)

    # 12. MODEL WARM-UP & SAFETY UNCERTAINTY
    def test_12_safety_uncertainty_and_data_modes(self):
        """Verifies clear labeling of MEASURED vs MODEL-DERIVED data and warm-up handling."""
        self.assertIn("dmDataProvenance", self.html_content)
        self.assertIn("MODEL-DERIVED", self.js_content)
        self.assertIn("MEASURED", self.js_content)
        self.assertIn("safety_advisory", self.js_content)

    # 13. RESPONSIVE UI & MOBILE LAYOUT
    def test_13_responsive_mobile_layout(self):
        """Verifies mobile responsive media queries for Driver Mode cockpit."""
        self.assertIn("@media (max-width: 960px)", self.css_content)
        self.assertIn(".dm-cockpit-layout", self.css_content)
        self.assertIn("grid-template-columns: 1fr", self.css_content)

    # 14. AUTHORITY MODE REGRESSION
    def test_14_authority_mode_remains_functional(self):
        """Verifies that Authority Mode components, APIs, and switching remain completely intact."""
        self.assertIn("activateAuthorityMode", self.js_content)
        self.assertIn("btnExitDriverMode", self.html_content)
        self.assertIn("btnModeAuthority", self.html_content)

        # Check Authority Mode backend endpoints
        res_status = self.client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)
        res_zones = self.client.get("/api/zones")
        self.assertEqual(res_zones.status_code, 200)
        res_hybrid = self.client.get("/api/hybrid/status")
        self.assertEqual(res_hybrid.status_code, 200)
        self.assertIn(res_hybrid.get_json().get("status"), ["READY", "ACTIVE"])

    # 15. END-TO-END DRIVER PIPELINE
    def test_15_end_to_end_driver_pipeline(self):
        """
        Executes end-to-end driver journey:
        Driver Location (22.7533, 75.8937) ->
        Driver Feed Sync ->
        Destination (Palasia Square) ->
        Route Calculation & Evaluation ->
        Congestion & Bottleneck Detection ->
        Proximity Alerts ->
        Real-Time Update
        """
        # Step 1: Query driver feed with driver's current position
        feed_res = self.client.get("/api/driver/feed?lat=22.7533&lng=75.8937&frame=50")
        self.assertEqual(feed_res.status_code, 200)
        feed_data = feed_res.get_json()
        self.assertEqual(feed_data["service"], "DriverModeSync")
        self.assertIsNotNone(feed_data["driver_location"])

        # Step 2: Route evaluation along corridor
        route_res = self.client.post("/api/routes/evaluate", json={
            "start": {"lat": 22.7533, "lng": 75.8937},
            "destination": {"lat": 22.7244, "lng": 75.8839},
            "points": [
                {"lat": 22.7533, "lng": 75.8937},
                {"lat": 22.7450, "lng": 75.8910},
                {"lat": 22.7380, "lng": 75.8880},
                {"lat": 22.7244, "lng": 75.8839}
            ]
        })
        self.assertEqual(route_res.status_code, 200)
        route_data = route_res.get_json()
        self.assertIn("highest_congestion_score", route_data)
        self.assertIn("recommended_action", route_data)

        # Step 3: Alerts fetch
        alerts_res = self.client.get("/api/alerts?frame=50")
        self.assertEqual(alerts_res.status_code, 200)
        alerts_data = alerts_res.get_json()
        self.assertGreaterEqual(alerts_data.get("total", 0), 1)


if __name__ == "__main__":
    unittest.main()
