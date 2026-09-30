import io
import time
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # el servidor nunca abre ventanas de gráficas

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from compare_opt import BRUTE_CALL_LIMIT, compare, optMemoized
from jobs import computeP, generateJobs, weightedSchedule

SIZES = list(range(5, 61, 5)) + [80, 100, 200, 500, 1000]
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Ecuación de Bellman: planificación de intervalos ponderados")


def jobsFor(n, seed):
    # Los mismos trabajos que compare() construye para esta n y semilla
    return generateJobs(n=n, max_time=max(18, n), seed=seed)


def callCounts(n, p):
    """cnt[j] = cuántas veces la fuerza bruta llama a OPT(j) al resolver OPT(n)."""
    cnt = [0] * (n + 1)
    cnt[n] = 1
    for j in range(n, 0, -1):
        cnt[p[j]] += cnt[j]
        cnt[j - 1] += cnt[j]
    return cnt


def memoTable(jobs, p):
    """M[j] = OPT(j) para cada j, llenado por la recursión memoizada."""
    memo = [None] * (len(jobs) + 1)
    optMemoized(len(jobs), jobs, p, memo, [0])
    return [0] + memo[1:]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/compare")
def apiCompare(seed: int = Query(0, ge=0), limit: int = Query(BRUTE_CALL_LIMIT, ge=1_000, le=30_000_000)):
    t0 = time.perf_counter()
    rows = compare(SIZES, seed=seed, limit=limit)
    elapsed = time.perf_counter() - t0

    out = []
    ns_per_call = None
    for n, b_calls, b_time, m_calls, m_time in rows:
        jobs = jobsFor(n, seed)
        p = computeP(jobs)
        total_calls = sum(callCounts(n, p))
        if b_calls is not None:
            ns_per_call = b_time * 1e9 / b_calls
        est_ms = None
        if b_calls is None and ns_per_call is not None:
            est_ms = total_calls * ns_per_call / 1e6
        out.append({
            "n": n,
            "value": weightedSchedule(jobs)[0],
            "bruteCalls": b_calls,
            "bruteMs": None if b_time is None else b_time * 1000,
            "memoCalls": m_calls,
            "memoMs": m_time * 1000,
            "totalCalls": str(total_calls),  # puede pasar de 2^53, se envía como texto
            "estMs": est_ms,
        })
    return {"seed": seed, "limit": limit, "elapsedMs": elapsed * 1000, "rows": out}


@app.get("/api/jobs")
def apiJobs(n: int = Query(10, ge=1, le=1000), seed: int = Query(0, ge=0)):
    jobs = jobsFor(n, seed)
    p = computeP(jobs)
    value, chosen = weightedSchedule(jobs)
    return {
        "n": n,
        "seed": seed,
        "jobs": [{"name": name, "weight": w, "start": s, "finish": f} for name, w, s, f in jobs],
        "p": p,
        "M": memoTable(jobs, p),
        "value": value,
        "chosen": chosen,
        "callCounts": [str(c) for c in callCounts(n, p)],
    }


class _Collect(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test)


@app.post("/api/tests")
def apiTests():
    suite = unittest.defaultTestLoader.loadTestsFromName("test_jobs")
    stream = io.StringIO()
    t0 = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, resultclass=_Collect, verbosity=0).run(suite)
    elapsed = time.perf_counter() - t0

    def entry(test, status, detail=""):
        cls, _, name = test.id().rpartition(".")
        return {"class": cls.rsplit(".", 1)[-1], "name": name, "status": status, "detail": detail}

    tests = ([entry(t, "passed") for t in result.passed]
             + [entry(t, "failed", tb) for t, tb in result.failures]
             + [entry(t, "error", tb) for t, tb in result.errors]
             + [entry(t, "skipped", why) for t, why in result.skipped])
    tests.sort(key=lambda e: (e["class"], e["name"]))
    return {
        "ran": result.testsRun,
        "passed": len(result.passed),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "elapsedMs": elapsed * 1000,
        "tests": tests,
    }
