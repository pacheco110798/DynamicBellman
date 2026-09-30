import io
import math
import threading
import time
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # el servidor nunca abre ventanas de gráficas

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from compare_opt import BRUTE_CALL_LIMIT, compare, optBruteForce, optMemoized, runBruteForce, runMemoized
from jobs import computeP, generateJobs, bestSchedule

SIZES = list(range(5, 61, 5)) + [80, 100, 200, 500, 1000]
STATIC = Path(__file__).parent / "static"

EXEC_N, EXEC_SEED = 45, 0  # n inicial de la ejecución de fuerza bruta de la pestaña "Ejecución"
EXEC_MAX_N = 60  # con n = 60 ya tarda unos 3 minutos y no se puede cancelar
SAMPLE_EVERY = 0.02  # segundos entre muestras de progreso

class Execution:
    """Fuerza bruta de OPT(n): se inicia desde la página y solo puede haber una corriendo a la vez."""

    def __init__(self):
        self.lock = threading.Lock()
        self.n = EXEC_N
        self._reset()
        self.status = "idle"

    def _reset(self):
        self.status = "pending"
        self.calls = [0]  # optBruteForce lo incrementa, el muestreador lo lee
        self.total = None
        self.startedAt = None
        self.seconds = None
        self.samples = []  # (segundos desde el inicio, llamadas)
        self.value = None
        self.chosen = None
        self.error = None

    def start(self, n=EXEC_N):
        """Inicia una ejecución para n trabajos; regresa False si ya hay una en curso."""
        with self.lock:
            if self.status in ("pending", "running"):
                return False
            self._reset()
            self.n = n
        threading.Thread(target=self._run, name="ejecucion-opt", daemon=True).start()
        return True

    def _run(self):
        try:
            jobs = jobsFor(self.n, EXEC_SEED)
            p = computeP(jobs)
            self.total = sum(callCounts(self.n, p))
            self.startedAt = time.time()
            t0 = time.perf_counter()
            self.samples.append((0.0, 0))
            self.status = "running"
            done = threading.Event()
            threading.Thread(target=self._sample, args=(t0, done), daemon=True).start()
            try:
                self.value = optBruteForce(self.n, jobs, p, self.calls, limit=math.inf)
            finally:
                done.set()
            self.seconds = time.perf_counter() - t0
            self.samples.append((self.seconds, self.calls[0]))
            self.chosen = bestSchedule(jobs)[1]
            self.status = "done"
        except Exception as e:
            self.error = repr(e)
            self.status = "error"

    def _sample(self, t0, done):
        while not done.wait(SAMPLE_EVERY):
            self.samples.append((time.perf_counter() - t0, self.calls[0]))

    def snapshot(self):
        elapsed = self.seconds
        if self.status == "running":
            elapsed = time.time() - self.startedAt
        return {
            "n": self.n,
            "maxN": EXEC_MAX_N,
            "seed": EXEC_SEED,
            "status": self.status,
            "calls": self.calls[0],
            "total": self.total,
            "startedAt": self.startedAt,
            "elapsed": elapsed,
            "samples": self.samples[::math.ceil(len(self.samples) / 300) or 1] + self.samples[-1:],  # máximo ~300
            "value": self.value,
            "chosen": self.chosen,
            "error": self.error,
        }


execution = Execution()


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
    # Sin caché, para que el navegador siempre cargue la versión actual de la página
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


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
            "value": bestSchedule(jobs)[0],
            "bruteCalls": b_calls,
            "bruteMs": None if b_time is None else b_time * 1000,
            "memoCalls": m_calls,
            "memoMs": m_time * 1000,
            "totalCalls": str(total_calls),  # puede pasar de 2^53, se envía como texto
            "estMs": est_ms,
        })
    return {"seed": seed, "limit": limit, "elapsedMs": elapsed * 1000, "rows": out}


_ms_per_call = None


def msPerBruteCall():
    """Tiempo por llamada de la fuerza bruta, medido una vez con un caso chico."""
    global _ms_per_call
    if _ms_per_call is None:
        sample = jobsFor(25, 0)
        _, calls, seconds = runBruteForce(sample, computeP(sample))
        _ms_per_call = seconds * 1000 / calls
    return _ms_per_call


def exerciseStats(jobs, p, limit):
    """Llamadas y tiempos del ejercicio actual: la fuerza bruta se mide si cabe en el límite, si no se estima."""
    total_calls = sum(callCounts(len(jobs), p))
    _, memo_calls, memo_time = runMemoized(jobs, p)
    brute_ms = est_ms = None
    if total_calls <= limit:
        brute_ms = runBruteForce(jobs, p, limit)[2] * 1000
    else:
        try:
            est_ms = total_calls * msPerBruteCall()
        except OverflowError:
            est_ms = None
    return {
        "bruteCalls": str(total_calls),  # puede pasar de 2^53, se envía como texto
        "bruteMs": brute_ms,
        "estMs": est_ms,
        "memoCalls": memo_calls,
        "memoMs": memo_time * 1000,
    }


@app.get("/api/jobs")
def apiJobs(n: int = Query(10, ge=1, le=1000), seed: int = Query(0, ge=0),
            limit: int = Query(BRUTE_CALL_LIMIT, ge=1_000, le=30_000_000)):
    jobs = jobsFor(n, seed)
    p = computeP(jobs)
    value, chosen = bestSchedule(jobs)
    return {
        "n": n,
        "seed": seed,
        "jobs": [{"name": name, "weight": w, "start": s, "finish": f} for name, w, s, f in jobs],
        "p": p,
        "M": memoTable(jobs, p),
        "value": value,
        "chosen": chosen,
        "callCounts": [str(c) for c in callCounts(n, p)],
        "stats": exerciseStats(jobs, p, limit),
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
        # Títulos en español: el docstring de la prueba y el de su clase (si no hay, el nombre en código)
        group = (type(test).__doc__ or "").strip() or cls.rsplit(".", 1)[-1]
        return {"class": cls.rsplit(".", 1)[-1], "name": name, "title": test.shortDescription() or name,
                "group": group, "status": status, "detail": detail}

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


@app.get("/api/execution")
def apiExecution():
    return execution.snapshot()


@app.post("/api/execution/start")
def apiExecutionStart(n: int = Query(EXEC_N, ge=1, le=EXEC_MAX_N)):
    # Solo una ejecución a la vez: si ya hay una en curso se rechaza con 409
    if not execution.start(n):
        return JSONResponse(status_code=409, content={"detail": "Ya hay una ejecución en curso", **execution.snapshot()})
    return execution.snapshot()
