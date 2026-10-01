"""
Verification Test Suite — Global Indian Location Search API & Coverage Rules
=============================================================================
Automated test suite verifying:
1. Input validation & error states (missing query, empty query, short query)
2. Arbitrary Indian location resolution: "MG Road, Katihar, Bihar"
3. Major Indian cities: Bhopal, Delhi, Mumbai, Bengaluru
4. Municipal sensor proximity evaluation (Indore vs Unmonitored locations)
5. Traffic Data Rule enforcement (has_traffic_data=False outside sensor range)
6. Non-existent location graceful handling
"""

import os
import sys
import json
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app import app


class TestLocationSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_01_search_missing_query(self):
        """GET /api/location/search without 'q' parameter should return 400 Bad Request."""
        res = self.client.get("/api/location/search")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_02_search_short_query(self):
        """GET /api/location/search with query < 2 chars should return count: 0."""
        res = self.client.get("/api/location/search?q=a")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("count"), 0)
        self.assertEqual(len(data.get("results")), 0)

    def test_03_search_mg_road_katihar_bihar(self):
        """
        Searches 'MG Road, Katihar, Bihar'.
        Verifies:
        - HTTP 200
        - Successfully resolves to Katihar, Bihar coordinates
        - Enforces Traffic Data Rule: has_traffic_data is False (outside sensor range)
        - Reports CCTV and traffic coverage as 'Not available'
        """
        res = self.client.get("/api/location/search?q=MG+Road%2C+Katihar%2C+Bihar")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertGreater(data.get("count", 0), 0)

        results = data.get("results", [])
        self.assertTrue(len(results) > 0)
        top = results[0]

        # Verify latitude and longitude are in Katihar / Bihar region (~25.5° N, ~87.5° E)
        self.assertAlmostEqual(top["latitude"], 25.54, delta=0.5)
        self.assertAlmostEqual(top["longitude"], 87.56, delta=0.5)

        # Verify Traffic Data Rule enforcement
        self.assertFalse(top["has_traffic_data"])
        self.assertEqual(top["cctv_status"], "Not available")
        self.assertEqual(top["traffic_status"], "Not available")
        self.assertEqual(top["model_coverage"], "Not available")
        self.assertIn("unavailable", top["coverage_message"].lower())

    def test_04_search_bhopal_outside_coverage(self):
        """Searches 'Bhopal' and verifies coords in MP (~23.25° N, ~77.40° E) with no fake data."""
        res = self.client.get("/api/location/search?q=Bhopal")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreater(data.get("count", 0), 0)

        top = data["results"][0]
        self.assertAlmostEqual(top["latitude"], 23.25, delta=0.5)
        self.assertAlmostEqual(top["longitude"], 77.40, delta=0.5)
        self.assertFalse(top["has_traffic_data"])
        self.assertEqual(top["cctv_status"], "Not available")

    def test_05_search_indore_monitored_sector(self):
        """Searches 'Vijay Nagar, Indore' and verifies within sensor range (has_traffic_data=True)."""
        res = self.client.get("/api/location/search?q=Vijay+Nagar%2C+Indore")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreater(data.get("count", 0), 0)

        # Among results, at least one should match active sensor in Vijay Nagar
        monitored_matches = [r for r in data["results"] if r.get("has_traffic_data")]
        self.assertTrue(len(monitored_matches) > 0)
        m = monitored_matches[0]
        self.assertTrue(m["has_traffic_data"])
        self.assertIsNotNone(m.get("closest_camera_id"))

    def test_06_search_delhi_mumbai_bengaluru(self):
        """Verifies Delhi, Mumbai, and Bengaluru return valid locations with unmonitored status."""
        cases = [
            ("Delhi", 28.6, 77.2),
            ("Mumbai", 19.0, 72.8),
            ("Bengaluru", 12.9, 77.5)
        ]
        for query, exp_lat, exp_lng in cases:
            with self.subTest(query=query):
                res = self.client.get(f"/api/location/search?q={query}")
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertGreater(data.get("count", 0), 0)
                top = data["results"][0]
                self.assertAlmostEqual(top["latitude"], exp_lat, delta=1.0)
                self.assertAlmostEqual(top["longitude"], exp_lng, delta=1.0)
                self.assertFalse(top["has_traffic_data"])
                self.assertEqual(top["cctv_status"], "Not available")

    def test_07_search_non_existent_location(self):
        """Verifies non-existent random query returns 200 with count 0 and empty results."""
        res = self.client.get("/api/location/search?q=xyzqwertynonexistentcity12345")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("count"), 0)
        self.assertEqual(len(data.get("results")), 0)


if __name__ == "__main__":
    unittest.main()
