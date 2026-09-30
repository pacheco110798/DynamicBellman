import sys
import time

import matplotlib.pyplot as plt

from jobs import computeP, generateJobs

sys.setrecursionlimit(10_000)

BRUTE_CALL_LIMIT = 2_000_000  # una ejecución de fuerza bruta se aborta después de estas llamadas


class TooManyCalls(Exception):
    pass


def optBruteForce(j, jobs, p, calls, limit=BRUTE_CALL_LIMIT):
    calls[0] += 1
    if calls[0] > limit:
        raise TooManyCalls
    if j == 0:
        return 0
    return max(jobs[j - 1][1] + optBruteForce(p[j], jobs, p, calls, limit),
               optBruteForce(j - 1, jobs, p, calls, limit))


def optMemoized(j, jobs, p, memo, calls):
    calls[0] += 1
    if j == 0:
        return 0
    if memo[j] is None:
        memo[j] = max(jobs[j - 1][1] + optMemoized(p[j], jobs, p, memo, calls),
                      optMemoized(j - 1, jobs, p, memo, calls))
    return memo[j]


def runBruteForce(jobs, p, limit=BRUTE_CALL_LIMIT):
    calls = [0]
    t0 = time.perf_counter()
    try:
        value = optBruteForce(len(jobs), jobs, p, calls, limit)
    except TooManyCalls:
        return None
    return value, calls[0], time.perf_counter() - t0


def runMemoized(jobs, p):
    calls = [0]
    memo = [None] * (len(jobs) + 1)
    t0 = time.perf_counter()
    value = optMemoized(len(jobs), jobs, p, memo, calls)
    return value, calls[0], time.perf_counter() - t0


def compare(ns, seed=0, limit=BRUTE_CALL_LIMIT):
    rows = []
    brute_enabled = True
    for n in ns:
        jobs = generateJobs(n=n, max_time=max(18, n), seed=seed)
        p = computeP(jobs)

        memo_value, memo_calls, memo_time = runMemoized(jobs, p)

        brute_calls = brute_time = None
        if brute_enabled:
            result = runBruteForce(jobs, p, limit)
            if result is None:
                brute_enabled = False
            else:
                brute_value, brute_calls, brute_time = result
                assert brute_value == memo_value, f"los resultados difieren para n={n}"

        rows.append((n, brute_calls, brute_time, memo_calls, memo_time))
    return rows


def printTable(rows):
    print(f"{'n':>5} | {'llam. bruta':>12} {'tiempo bruta':>12} | {'llam. memo':>10} {'tiem. memo':>10} | aceleración")
    print("-" * 72)
    for n, b_calls, b_time, m_calls, m_time in rows:
        if b_calls is None:
            brute = f"{'—':>12} {'muy lento':>12}"
            speedup = ""
        else:
            brute = f"{b_calls:>12,} {b_time * 1000:>10.2f}ms"
            speedup = f"{b_time / m_time:>8.0f}x" if m_time else ""
        print(f"{n:>5} | {brute} | {m_calls:>10,} {m_time * 1000:>8.3f}ms | {speedup}")


def plotComparison(rows, save_path=None, show=True):
    ns = [r[0] for r in rows]
    brute = [r for r in rows if r[1] is not None]

    fig, (ax_calls, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    fig.canvas.manager.set_window_title("Fuerza bruta vs memoizado")

    ax_calls.plot([r[0] for r in brute], [r[1] for r in brute], "o-", color="tab:red", label="Fuerza bruta")
    ax_calls.plot(ns, [r[3] for r in rows], "o-", color="tab:blue", label="Memoizado")
    ax_calls.set_yscale("log")
    ax_calls.set_xlabel("n (trabajos)")
    ax_calls.set_ylabel("llamadas a OPT (escala log)")
    ax_calls.set_title("Número de llamadas")

    ax_time.plot([r[0] for r in brute], [r[2] * 1000 for r in brute], "o-", color="tab:red", label="Fuerza bruta")
    ax_time.plot(ns, [r[4] * 1000 for r in rows], "o-", color="tab:blue", label="Memoizado")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("n (trabajos)")
    ax_time.set_ylabel("tiempo en ms (escala log)")
    ax_time.set_title("Tiempo de ejecución")

    for ax in (ax_calls, ax_time):
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        if len(brute) < len(rows):
            ax.axvline(brute[-1][0], color="tab:red", linestyle=":", alpha=0.6)
            ax.text(brute[-1][0], ax.get_ylim()[1], " fuerza bruta detenida", color="tab:red",
                    va="top", fontsize=9)

    fig.suptitle("OPT(j) = max(w_j + OPT(p(j)), OPT(j-1)): fuerza bruta es exponencial, memoizado es lineal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()


if __name__ == "__main__":
    NS = list(range(5, 41, 5)) + [60, 80, 100, 200, 500, 1000]
    ROWS = compare(NS, seed=0)
    printTable(ROWS)
    plotComparison(ROWS, save_path="brute_vs_memo.png")
