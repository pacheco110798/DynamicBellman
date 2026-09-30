import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from itertools import combinations

from compare_opt import compare, runBruteForce, runMemoized
from jobs import (computeP, SORTED_BY_FINISH_ASC, buildJobTree, countNodes, findPOfJob, fitDepth, generateJobs,
                  plotJobs, plotJobTree, weightedSchedule)


def bruteForcePOfJob(jobname, jobs):
    start = next(job[2] for job in jobs if job[0] == jobname)
    result = -1
    for job in jobs:
        if job[3] <= start:
            result = job[0]
    return result


def compatible(jobs):
    """True si ningún par de trabajos se traslapa (que se toquen en los extremos está bien)."""
    ordered = sorted(jobs, key=lambda job: job[2])
    return all(a[3] <= b[2] for a, b in zip(ordered, ordered[1:]))


def bruteForceSchedule(jobs):
    """Prueba todos los subconjuntos de trabajos y devuelve el mejor peso total de uno compatible."""
    best = 0
    for size in range(1, len(jobs) + 1):
        for subset in combinations(jobs, size):
            if compatible(subset):
                best = max(best, sum(job[1] for job in subset))
    return best


class TestGenerateJobs(unittest.TestCase):
    def test_returns_n_jobs(self):
        self.assertEqual(len(generateJobs(n=12, seed=0)), 12)

    def test_zero_jobs(self):
        self.assertEqual(generateJobs(n=0, seed=0), [])

    def test_job_structure(self):
        for job in generateJobs(n=50, seed=1):
            self.assertEqual(len(job), 4)
            name, weight, start, finish = job
            self.assertIsInstance(name, str)
            self.assertIsInstance(weight, int)
            self.assertIsInstance(start, int)
            self.assertIsInstance(finish, int)

    def test_names_are_sequential(self):
        names = [job[0] for job in generateJobs(n=5, seed=2)]
        self.assertEqual(names, ["J1", "J2", "J3", "J4", "J5"])

    def test_names_follow_finish_order(self):
        for seed in range(20):
            jobs = generateJobs(n=30, seed=seed)
            finishes = [job[3] for job in jobs]
            self.assertEqual(finishes, sorted(finishes))
            self.assertEqual(jobs[-1][0], "J30")

    def test_values_within_bounds(self):
        max_time, max_weight, min_d, max_d = 20, 15, 2, 6
        jobs = generateJobs(n=200, max_time=max_time, max_weight=max_weight,
                            min_duration=min_d, max_duration=max_d, seed=3)
        for name, weight, start, finish in jobs:
            self.assertTrue(1 <= weight <= max_weight)
            self.assertTrue(0 <= start < finish <= max_time)
            self.assertTrue(min_d <= finish - start <= max_d)

    def test_same_seed_same_jobs(self):
        self.assertEqual(generateJobs(n=10, seed=42), generateJobs(n=10, seed=42))

    def test_different_seed_different_jobs(self):
        self.assertNotEqual(generateJobs(n=10, seed=1), generateJobs(n=10, seed=2))


class TestFindPOfJobFixed(unittest.TestCase):

    def test_known_values(self):
        expected = {
            "J1": -1,    # empieza en 1, nada termina antes
            "J2": -1,    # empieza en 3
            "J3": -1,    # empieza en 0
            "J4": "J1",  # empieza en 4, J1 termina en 4
            "J5": -1,    # empieza en 3
            "J6": "J2",  # empieza en 5, J2 termina en 5
            "J7": "J3",  # empieza en 6, J3 termina en 6
            "J8": "J1",  # empieza en 4
        }
        for job, p in expected.items():
            with self.subTest(job=job):
                self.assertEqual(findPOfJob(job), p)

    def test_unknown_job(self):
        self.assertEqual(findPOfJob("J99"), -1)

    def test_default_list_is_sorted_by_finish(self):
        finishes = [job[3] for job in SORTED_BY_FINISH_ASC]
        self.assertEqual(finishes, sorted(finishes))


class TestFindPOfJobCustomList(unittest.TestCase):
    def test_single_job(self):
        self.assertEqual(findPOfJob("A", [("A", 1, 0, 3)]), -1)

    def test_empty_list(self):
        self.assertEqual(findPOfJob("A", []), -1)

    def test_touching_jobs_are_compatible(self):
        jobs = [("A", 1, 0, 3), ("B", 1, 3, 5)]
        self.assertEqual(findPOfJob("B", jobs), "A")

    def test_overlapping_jobs_are_not_compatible(self):
        jobs = [("A", 1, 0, 4), ("B", 1, 3, 5)]
        self.assertEqual(findPOfJob("B", jobs), -1)

    def test_picks_latest_compatible(self):
        jobs = [("A", 1, 0, 1), ("B", 1, 0, 2), ("C", 1, 1, 3), ("D", 1, 3, 6)]
        self.assertEqual(findPOfJob("D", jobs), "C")


class TestFindPOfJobRandom(unittest.TestCase):
    def test_matches_brute_force(self):
        for seed in range(100):
            jobs = sorted(generateJobs(n=15, seed=seed), key=lambda job: job[3])
            for job in jobs:
                with self.subTest(seed=seed, job=job[0]):
                    self.assertEqual(findPOfJob(job[0], jobs), bruteForcePOfJob(job[0], jobs))

    def test_result_finishes_before_start(self):
        for seed in range(50):
            jobs = sorted(generateJobs(n=10, seed=seed), key=lambda job: job[3])
            by_name = {job[0]: job for job in jobs}
            for job in jobs:
                p = findPOfJob(job[0], jobs)
                if p != -1:
                    self.assertLessEqual(by_name[p][3], job[2])


class TestWeightedSchedule(unittest.TestCase):
    def test_ujobs(self):
        self.assertEqual(weightedSchedule(), (17, ["J2", "J7"]))

    def test_empty_list(self):
        self.assertEqual(weightedSchedule([]), (0, []))

    def test_single_job(self):
        self.assertEqual(weightedSchedule([("A", 4, 0, 3)]), (4, ["A"]))

    def test_heavy_job_beats_two_light_ones(self):
        jobs = [("A", 2, 0, 2), ("B", 2, 2, 4), ("C", 10, 0, 4)]
        self.assertEqual(weightedSchedule(jobs), (10, ["C"]))

    def test_two_light_jobs_beat_one_heavy(self):
        jobs = [("A", 6, 0, 2), ("B", 6, 2, 4), ("C", 10, 0, 4)]
        self.assertEqual(weightedSchedule(jobs), (12, ["A", "B"]))

    def test_matches_brute_force(self):
        for seed in range(150):
            jobs = sorted(generateJobs(n=10, seed=seed), key=lambda job: job[3])
            with self.subTest(seed=seed):
                best, _ = weightedSchedule(jobs)
                self.assertEqual(best, bruteForceSchedule(jobs))

    def test_chosen_jobs_are_valid(self):
        for seed in range(100):
            jobs = sorted(generateJobs(n=30, seed=seed), key=lambda job: job[3])
            by_name = {job[0]: job for job in jobs}
            best, chosen = weightedSchedule(jobs)
            chosen_jobs = [by_name[name] for name in chosen]
            with self.subTest(seed=seed):
                self.assertTrue(compatible(chosen_jobs))
                self.assertEqual(sum(job[1] for job in chosen_jobs), best)


class TestJobTree(unittest.TestCase):
    def test_root_is_last_job(self):
        root = buildJobTree()
        self.assertEqual((root.j, root.name), (8, "J8"))

    def test_ujobs_tree(self):
        root = buildJobTree()
        self.assertEqual(root.value, 17)
        self.assertEqual(countNodes(root), 31)
        self.assertEqual((root.take.j, root.skip.j), (1, 7))  # p(8) = 1

    def test_empty_list(self):
        root = buildJobTree([])
        self.assertEqual((root.j, root.value), (0, 0))
        self.assertEqual(countNodes(root), 1)

    def test_every_node_is_full(self):
        """Los nodos internos tienen exactamente dos hijos, las hojas OPT(0) no tienen ninguno."""
        def check(node):
            if node.j == 0:
                self.assertIsNone(node.take)
                self.assertIsNone(node.skip)
            else:
                self.assertEqual(node.skip.j, node.j - 1)
                self.assertLess(node.take.j, node.j)
                check(node.take)
                check(node.skip)
        check(buildJobTree())

    def test_root_value_matches_weighted_schedule(self):
        for seed in range(50):
            jobs = sorted(generateJobs(n=12, seed=seed), key=lambda job: job[3])
            with self.subTest(seed=seed):
                self.assertEqual(buildJobTree(jobs).value, weightedSchedule(jobs)[0])

    @patch("jobs.plt.show")
    def test_plot_tree(self, mock_show):
        plotJobTree(buildJobTree())
        mock_show.assert_called_once()

    @patch("jobs.plt.show")
    def test_plot_big_tree_is_cut(self, mock_show):
        jobs = sorted(generateJobs(n=20, seed=0), key=lambda job: job[3])
        root = buildJobTree(jobs)
        depth, full_depth = fitDepth(root)
        self.assertLess(depth, full_depth)
        plotJobTree(root)
        mock_show.assert_called_once()

    def test_fit_depth_small_tree_is_complete(self):
        depth, full_depth = fitDepth(buildJobTree())
        self.assertEqual(depth, full_depth)


class TestPlotJobs(unittest.TestCase):
    @patch("jobs.plt.show")
    def test_plot_with_schedule_small(self, mock_show):
        _, chosen = weightedSchedule()
        plotJobs("J7", findPOfJob("J7"), chosen=chosen)
        mock_show.assert_called_once()

    @patch("jobs.plt.show")
    def test_plot_with_schedule_large(self, mock_show):
        jobs = sorted(generateJobs(n=100, seed=7), key=lambda job: job[3])
        _, chosen = weightedSchedule(jobs)
        plotJobs(jobs[-1][0], findPOfJob(jobs[-1][0], jobs), jobs, chosen)
        mock_show.assert_called_once()


class TestBruteVsMemo(unittest.TestCase):
    def test_same_value_as_weighted_schedule(self):
        for seed in range(50):
            jobs = generateJobs(n=15, seed=seed)
            p = computeP(jobs)
            with self.subTest(seed=seed):
                best = weightedSchedule(jobs)[0]
                self.assertEqual(runBruteForce(jobs, p)[0], best)
                self.assertEqual(runMemoized(jobs, p)[0], best)

    def test_brute_force_calls_equal_tree_size(self):
        for seed in range(20):
            jobs = generateJobs(n=12, seed=seed)
            with self.subTest(seed=seed):
                self.assertEqual(runBruteForce(jobs, computeP(jobs))[1], countNodes(buildJobTree(jobs)))

    def test_memoized_calls_are_linear(self):
        for n in (1, 10, 100, 1000):
            jobs = generateJobs(n=n, seed=0)
            with self.subTest(n=n):
                self.assertEqual(runMemoized(jobs, computeP(jobs))[1], 2 * n + 1)

    def test_compare_stops_brute_force(self):
        rows = compare([5, 10, 60], seed=0)
        self.assertIsNotNone(rows[0][1])
        self.assertIsNone(rows[-1][1])


if __name__ == "__main__":
    unittest.main()
