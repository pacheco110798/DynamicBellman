import unittest
from itertools import combinations

from compare_opt import compare, runBruteForce, runMemoized
from jobs import (computeP, SORTED_BY_FINISH_ASC, findPOfJob, generateJobs,
                  bestSchedule)

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


class TestFindPOfJobFixed(unittest.TestCase):
    """p(j) con los 8 trabajos de ejemplo"""

    def test_known_values(self):
        """Cada trabajo de ejemplo tiene el p(j) esperado"""
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
        """Un trabajo que no existe regresa -1"""
        self.assertEqual(findPOfJob("J99"), -1)

    def test_default_list_is_sorted_by_finish(self):
        """La lista de ejemplo está ordenada por tiempo de término"""
        finishes = [job[3] for job in SORTED_BY_FINISH_ASC]
        self.assertEqual(finishes, sorted(finishes))


class TestFindPOfJobCustomList(unittest.TestCase):
    """p(j) con listas pequeñas hechas a mano"""

    def test_single_job(self):
        """Con un solo trabajo no hay p(j)"""
        self.assertEqual(findPOfJob("A", [("A", 1, 0, 3)]), -1)

    def test_empty_list(self):
        """Con una lista vacía no hay p(j)"""
        self.assertEqual(findPOfJob("A", []), -1)

    def test_touching_jobs_are_compatible(self):
        """Dos trabajos que solo se tocan en un extremo son compatibles"""
        jobs = [("A", 1, 0, 3), ("B", 1, 3, 5)]
        self.assertEqual(findPOfJob("B", jobs), "A")

    def test_overlapping_jobs_are_not_compatible(self):
        """Dos trabajos que se traslapan no son compatibles"""
        jobs = [("A", 1, 0, 4), ("B", 1, 3, 5)]
        self.assertEqual(findPOfJob("B", jobs), -1)

    def test_picks_latest_compatible(self):
        """p(j) es el último trabajo compatible, no cualquiera"""
        jobs = [("A", 1, 0, 1), ("B", 1, 0, 2), ("C", 1, 1, 3), ("D", 1, 3, 6)]
        self.assertEqual(findPOfJob("D", jobs), "C")


class TestFindPOfJobRandom(unittest.TestCase):
    """p(j) con trabajos aleatorios"""

    def test_matches_brute_force(self):
        """p(j) coincide con buscarlo trabajo por trabajo en 100 casos aleatorios"""
        for seed in range(100):
            jobs = sorted(generateJobs(n=15, seed=seed), key=lambda job: job[3])
            for job in jobs:
                with self.subTest(seed=seed, job=job[0]):
                    self.assertEqual(findPOfJob(job[0], jobs), bruteForcePOfJob(job[0], jobs))

    def test_result_finishes_before_start(self):
        """El trabajo p(j) siempre termina antes de que empiece j"""
        for seed in range(50):
            jobs = sorted(generateJobs(n=10, seed=seed), key=lambda job: job[3])
            by_name = {job[0]: job for job in jobs}
            for job in jobs:
                p = findPOfJob(job[0], jobs)
                if p != -1:
                    self.assertLessEqual(by_name[p][3], job[2])


class TestWeightedSchedule(unittest.TestCase):
    """Planificación óptima (bestSchedule)"""

    def test_ujobs(self):
        """Con los trabajos de ejemplo la ganancia es 17 usando J2 y J7"""
        self.assertEqual(bestSchedule(), (17, ["J2", "J7"]))

    def test_empty_list(self):
        """Sin trabajos la ganancia es 0"""
        self.assertEqual(bestSchedule([]), (0, []))

    def test_single_job(self):
        """Con un solo trabajo, ese trabajo es la solución"""
        self.assertEqual(bestSchedule([("A", 4, 0, 3)]), (4, ["A"]))

    def test_heavy_job_beats_two_light_ones(self):
        """Un trabajo pesado gana a dos ligeros que se le traslapan"""
        jobs = [("A", 2, 0, 2), ("B", 2, 2, 4), ("C", 10, 0, 4)]
        self.assertEqual(bestSchedule(jobs), (10, ["C"]))

    def test_two_light_jobs_beat_one_heavy(self):
        """Dos trabajos ligeros ganan a uno pesado si suman más"""
        jobs = [("A", 6, 0, 2), ("B", 6, 2, 4), ("C", 10, 0, 4)]
        self.assertEqual(bestSchedule(jobs), (12, ["A", "B"]))

    def test_matches_brute_force(self):
        """La ganancia coincide con probar todos los subconjuntos en 150 casos"""
        for seed in range(150):
            jobs = sorted(generateJobs(n=10, seed=seed), key=lambda job: job[3])
            with self.subTest(seed=seed):
                best, _ = bestSchedule(jobs)
                self.assertEqual(best, bruteForceSchedule(jobs))

    def test_chosen_jobs_are_valid(self):
        """Los trabajos elegidos no se traslapan y suman la ganancia óptima"""
        for seed in range(100):
            jobs = sorted(generateJobs(n=30, seed=seed), key=lambda job: job[3])
            by_name = {job[0]: job for job in jobs}
            best, chosen = bestSchedule(jobs)
            chosen_jobs = [by_name[name] for name in chosen]
            with self.subTest(seed=seed):
                self.assertTrue(compatible(chosen_jobs))
                self.assertEqual(sum(job[1] for job in chosen_jobs), best)


class TestBruteVsMemo(unittest.TestCase):
    """Fuerza bruta vs memoizado"""

    def test_same_value_as_weighted_schedule(self):
        """Fuerza bruta y memoizado dan la misma ganancia que bestSchedule"""
        for seed in range(50):
            jobs = generateJobs(n=15, seed=seed)
            p = computeP(jobs)
            with self.subTest(seed=seed):
                best = bestSchedule(jobs)[0]
                self.assertEqual(runBruteForce(jobs, p)[0], best)
                self.assertEqual(runMemoized(jobs, p)[0], best)

    def test_memoized_calls_are_linear(self):
        """El memoizado hace exactamente 2n + 1 llamadas"""
        for n in (1, 10, 100, 1000):
            jobs = generateJobs(n=n, seed=0)
            with self.subTest(n=n):
                self.assertEqual(runMemoized(jobs, computeP(jobs))[1], 2 * n + 1)

    def test_compare_stops_brute_force(self):
        """La comparación detiene la fuerza bruta cuando pasa el límite de llamadas"""
        rows = compare([5, 10, 60], seed=0)
        self.assertIsNotNone(rows[0][1])
        self.assertIsNone(rows[-1][1])


if __name__ == "__main__":
    unittest.main()
