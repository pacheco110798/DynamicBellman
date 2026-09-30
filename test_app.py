import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import SIZES, app, callCounts
from jobs import buildJobTree, computeP, countNodes, generateJobs, weightedSchedule

client = TestClient(app)


class TestIndex(unittest.TestCase):
    def test_serves_page(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("OPT(j)", res.text)


class TestCallCounts(unittest.TestCase):
    def test_total_equals_tree_size(self):
        for seed in range(20):
            jobs = generateJobs(n=12, seed=seed)
            with self.subTest(seed=seed):
                self.assertEqual(sum(callCounts(12, computeP(jobs))), countNodes(buildJobTree(jobs)))


class TestCompareEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = client.get("/api/compare", params={"seed": 0, "limit": 500_000}).json()

    def test_one_row_per_size(self):
        self.assertEqual([r["n"] for r in self.data["rows"]], SIZES)

    def test_memoized_calls_are_linear(self):
        for r in self.data["rows"]:
            self.assertEqual(r["memoCalls"], 2 * r["n"] + 1)

    def test_brute_force_stops_and_gets_estimate(self):
        rows = self.data["rows"]
        self.assertIsNotNone(rows[0]["bruteCalls"])
        self.assertIsNone(rows[-1]["bruteCalls"])
        self.assertIsNotNone(rows[-1]["estMs"])

    def test_finished_brute_calls_match_exact_count(self):
        for r in self.data["rows"]:
            if r["bruteCalls"] is not None:
                self.assertEqual(r["bruteCalls"], int(r["totalCalls"]))

    def test_rejects_bad_seed(self):
        self.assertEqual(client.get("/api/compare", params={"seed": -1}).status_code, 422)


class TestJobsEndpoint(unittest.TestCase):
    def test_matches_weighted_schedule(self):
        for seed in range(10):
            data = client.get("/api/jobs", params={"n": 15, "seed": seed}).json()
            jobs = generateJobs(n=15, max_time=18, seed=seed)
            value, chosen = weightedSchedule(jobs)
            with self.subTest(seed=seed):
                self.assertEqual(data["value"], value)
                self.assertEqual(data["chosen"], chosen)
                self.assertEqual(data["p"], computeP(jobs))
                self.assertEqual(data["M"][-1], value)

    def test_opt_table_is_nondecreasing(self):
        M = client.get("/api/jobs", params={"n": 1000, "seed": 3}).json()["M"]
        self.assertEqual(M, sorted(M))

    def test_rejects_n_out_of_range(self):
        self.assertEqual(client.get("/api/jobs", params={"n": 0}).status_code, 422)
        self.assertEqual(client.get("/api/jobs", params={"n": 1001}).status_code, 422)


class TestTestsEndpoint(unittest.TestCase):
    @patch("unittest.defaultTestLoader.loadTestsFromName")
    def test_reports_results(self, mock_load):
        class Sample(unittest.TestCase):
            def test_ok(self):
                pass

            def test_bad(self):
                self.fail("boom")

        mock_load.return_value = unittest.defaultTestLoader.loadTestsFromTestCase(Sample)
        data = client.post("/api/tests").json()
        self.assertEqual((data["ran"], data["passed"], data["failed"]), (2, 1, 1))
        bad = next(t for t in data["tests"] if t["name"] == "test_bad")
        self.assertEqual(bad["status"], "failed")
        self.assertIn("boom", bad["detail"])


if __name__ == "__main__":
    unittest.main()
