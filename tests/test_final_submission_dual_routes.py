"""
Final Submission Test Suite: Dual Routing, Location Intelligence, and Policy Recommendation Engine
Verifies:
  1. /api/routes/compare (Shortest Route vs. Traffic-Aware Best Route)
  2. Strict zero-fabrication of traffic data for unmonitored regions (e.g. Katihar, Bihar)
  3. /api/authority/policy-recommendations with explicit "RECOMMENDATION ONLY — NOT REAL-ROAD ACTUATION"
  4. /api/authority/location-intelligence with complete provenance matrix
  5. SUMO digital twin integration and simulation safety markers
"""

import unittest
import json
from app import app


class TestFinalSubmissionDualRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    # 1. DUAL ROUTE COMPARISON (IN-COVERAGE)
    def test_01_dual_route_in_coverage(self):
        """Verifies dual route calculation when corridor is inside municipal sensor coverage."""
        payload = {
            "start": {"lat": 22.7533, "lng": 75.8937},
            "destination": {"lat": 22.7244, "lng": 75.8839, "name": "Palasia Square, Indore"}
        }
        res = self.client.post("/api/routes/compare", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "success")
        self.assertIn("shortest_route", data)
        self.assertIn("traffic_aware_route", data)
        self.assertIn("comparison", data)
        self.assertIn("coverage", data)

        shortest = data["shortest_route"]
        aware = data["traffic_aware_route"]

        # Shortest minimizes physical distance
        self.assertGreater(shortest["distance_km"], 0)
        self.assertIn("NO", shortest["traffic_optimization"])
        self.assertIn("path_coordinates", shortest)
        self.assertGreaterEqual(len(shortest["path_coordinates"]), 2)

        # Traffic-aware optimizes time and congestion
        self.assertGreater(aware["distance_km"], 0)
        self.assertIn("YES", aware["traffic_optimization"])
        self.assertIn("reason", aware)
        self.assertIn("policy_impact", aware)
        self.assertIn("path_coordinates", aware)

        # In-coverage: live traffic is available
        self.assertTrue(data["coverage"]["live_traffic_available"])
        self.assertEqual(data["coverage"]["status"], "FULL COVERAGE")

    # 2. DUAL ROUTE UNMONITORED REGION (ZERO FAKE DATA)
    def test_02_dual_route_unmonitored_region_no_fake_data(self):
        """Verifies that an unmonitored location (e.g. Katihar, Bihar) displays UNAVAILABLE instead of fake congestion."""
        payload = {
            "start": {"lat": 25.5398, "lng": 87.5721},
            "destination": {"lat": 25.5510, "lng": 87.5850, "name": "MG Road, Katihar, Bihar"}
        }
        res = self.client.post("/api/routes/compare", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "success")
        self.assertFalse(data["coverage"]["live_traffic_available"])
        self.assertIn("Traffic data unavailable", data["coverage"]["message"])

        # Congestion must NOT be fabricated
        self.assertEqual(data["shortest_route"]["congestion_level"], "UNAVAILABLE")
        self.assertIsNone(data["shortest_route"]["congestion_score"])
        self.assertEqual(data["traffic_aware_route"]["congestion_level"], "UNAVAILABLE")
        self.assertIsNone(data["traffic_aware_route"]["congestion_score"])

    # 3. AUTHORITY POLICY RECOMMENDATIONS
    def test_03_authority_policy_recommendations(self):
        """Verifies policy recommendation engine returns actionable guidance with non-actuation disclaimer."""
        res = self.client.get("/api/authority/policy-recommendations?frame=100")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "success")
        self.assertIn("condition", data)
        self.assertIn("recommendation", data)
        self.assertIn("reason", data)
        self.assertEqual(data["data_source"], "MEASURED / PREDICTED / DERIVED")
        self.assertEqual(data["actuation_status"], "RECOMMENDATION ONLY — NOT REAL-ROAD ACTUATION")
        self.assertIn("SIMULATED CONTROL ACTION", data["simulated_action_label"])
        self.assertIsInstance(data["recommendations"], list)
        self.assertGreater(len(data["recommendations"]), 0)

    # 4. LOCATION INTELLIGENCE (IN-COVERAGE)
    def test_04_location_intelligence_monitored(self):
        """Verifies location intelligence for municipal CCTV covered node."""
        res = self.client.get("/api/authority/location-intelligence?lat=22.7533&lng=75.8937&q=Vijay+Nagar")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["cctv_status"], "AVAILABLE (ONLINE)")
        self.assertEqual(data["live_traffic_status"], "AVAILABLE")
        self.assertEqual(data["location_status"], "MONITORED SURVEILLANCE NODE")
        self.assertEqual(data["provenance_matrix"]["project_cctv"], "AVAILABLE")
        self.assertEqual(data["provenance_matrix"]["live_project_traffic"], "AVAILABLE")

    # 5. LOCATION INTELLIGENCE (UNMONITORED INDIAN LOCATION)
    def test_05_location_intelligence_unmonitored_katihar(self):
        """Verifies location intelligence for arbitrary Indian location without CCTV (MG Road, Katihar, Bihar)."""
        res = self.client.get("/api/authority/location-intelligence?lat=25.5398&lng=87.5721&q=MG+Road,+Katihar,+Bihar")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["cctv_status"], "NOT AVAILABLE")
        self.assertEqual(data["live_traffic_status"], "UNAVAILABLE")
        self.assertEqual(data["project_traffic_coverage"], "NOT AVAILABLE")
        self.assertEqual(data["location_status"], "ROAD LOCATION IDENTIFIED")
        self.assertEqual(data["traffic_intelligence"], "LIMITED")
        self.assertIn("No connected CCTV or project traffic sensor", data["message"])
        self.assertIn("HISTORICAL / AVAILABLE DATA", data["typical_traffic_character"])

        matrix = data["provenance_matrix"]
        self.assertEqual(matrix["map"], "AVAILABLE")
        self.assertEqual(matrix["road_data"], "AVAILABLE")
        self.assertEqual(matrix["project_cctv"], "NOT AVAILABLE")
        self.assertEqual(matrix["live_project_traffic"], "UNAVAILABLE")
        self.assertEqual(matrix["model_prediction"], "UNAVAILABLE")
        self.assertEqual(matrix["simulation"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
