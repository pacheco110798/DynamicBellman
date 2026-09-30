import random

import matplotlib.pyplot as plt

ujobs = [
    #nombre, peso, inicio, fin
    ("J1", 5, 1, 4),
    ("J2", 8, 3, 5),
    ("J3", 3, 0, 6),
    ("J4", 10, 4, 7),
    ("J5", 6, 3, 8),
    ("J6", 4, 5, 9),
    ("J7", 9, 6, 10),
    ("J8", 7, 4, 11),
]

def generateJobs(n=8, max_time=18, max_weight=10, min_duration=1, max_duration=7, seed=None):
    rng = random.Random(seed)
    intervals = []
    for _ in range(n):
        duration = rng.randint(min_duration, max_duration)
        start = rng.randint(0, max_time - duration)
        weight = rng.randint(1, max_weight)
        intervals.append((weight, start, start + duration))
    # Se nombran los trabajos después de ordenarlos por tiempo de fin, así J1 termina primero y Jn al último
    intervals.sort(key=lambda job: (job[2], job[1]))
    return [(f"J{i}", weight, start, finish) for i, (weight, start, finish) in enumerate(intervals, 1)]


SORTED_BY_FINISH_ASC = sorted(ujobs, key=lambda job: job[3])
BAR_LIMIT = 40  # con más trabajos que esto se dibujan líneas delgadas en vez de barras con etiqueta

def plotJobs(job, pofjob, jobs=SORTED_BY_FINISH_ASC, chosen=(), show=True):
    chosen = set(chosen)
    names = [job_data[0] for job_data in jobs]
    weights = [job_data[1] for job_data in jobs]
    max_weight = max(weights)
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(jobs) <= BAR_LIMIT:
        # Dibuja barras horizontales
        for i, job_data in enumerate(reversed(jobs)):
            name, weight, start, finish = job_data
            color = plt.cm.viridis(weight / max_weight)
            if name == job or name == pofjob:
                color = "red"
            if name in chosen:
                # Planificación óptima: con achurado y borde grueso
                ax.barh(y=i, width=finish - start, left=start, color=color, edgecolor='black',
                        linewidth=2.5, hatch='//')
            else:
                ax.barh(y=i, width=finish - start, left=start, color=color, edgecolor='black')
            ax.text(start + (finish - start) / 2, i, name, ha='center', va='center', fontsize=10)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(list(reversed(names)))
    else:
        # Demasiados trabajos para barras: una línea delgada por trabajo, coloreada según su peso
        rows = list(reversed(jobs))
        ys = range(len(rows))
        ax.hlines(y=ys, xmin=[j[2] for j in rows], xmax=[j[3] for j in rows],
                  colors=[plt.cm.viridis(j[1] / max_weight) for j in rows], linewidth=1)
        # Planificación óptima como líneas negras gruesas
        chosen_rows = [(i, j) for i, j in enumerate(rows) if j[0] in chosen]
        if chosen_rows:
            ax.hlines(y=[i for i, _ in chosen_rows], xmin=[j[2] for _, j in chosen_rows],
                      xmax=[j[3] for _, j in chosen_rows], colors="black", linewidth=3, zorder=2,
                      label=f"Planificación óptima ({len(chosen_rows)} trabajos)")
        # Trabajos resaltados encima, más gruesos y nombrados en la leyenda
        for i, (name, weight, start, finish) in enumerate(rows):
            if name == job or name == pofjob:
                style = "red" if name == job else "orange"
                label = f"{name} [{start}, {finish})" + (" (p)" if name == pofjob else "")
                ax.hlines(y=i, xmin=start, xmax=finish, colors=style, linewidth=4, zorder=3, label=label)
                ax.plot([start, finish], [i, i], "|", color=style, markersize=12, zorder=3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
        ax.set_ylim(-1, len(rows))
        ax.set_yticks([])
        fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max_weight)),
                     ax=ax, label="Peso")

    # Etiquetas y título
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Trabajo")
    title = f"Trabajos (n={len(jobs)})"
    if chosen:
        total = sum(job_data[1] for job_data in jobs if job_data[0] in chosen)
        title += f" — peso óptimo {total}"
    ax.set_title(title)
    max_finish = max(18, max(job_data[3] for job_data in jobs))
    ax.set_xlim(0, max_finish + 1)
    # Líneas guía para comparar tiempos más fácil (máximo ~20 marcas)
    ax.set_xticks(range(0, max_finish + 1, max(1, max_finish // 20)))
    ax.grid(axis='x', linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    if show:
        plt.show()


def findPOfJob(jobname, jobs=SORTED_BY_FINISH_ASC):
    idx = next((i for i, v in enumerate(jobs) if v[0] == jobname), -1)

    if idx == -1:
        return -1

    startime = jobs[idx][2]
    finishes = [job[3] for job in jobs]
    low = 0
    high = len(finishes) - 1
    result = -1

    while low <= high:
        mid = low + (high - low) // 2
        mid_value = finishes[mid]

        if mid_value <= startime:
            result = jobs[mid][0]
            low = mid + 1
        else:
            high = mid - 1

    return result


def computeP(jobs=SORTED_BY_FINISH_ASC):
    index = {job[0]: i + 1 for i, job in enumerate(jobs)}  # nombre -> posición 1..n
    p = [0] * (len(jobs) + 1)
    for j in range(1, len(jobs) + 1):
        pj = findPOfJob(jobs[j - 1][0], jobs)
        p[j] = 0 if pj == -1 else index[pj]
    return p


def weightedSchedule(jobs=SORTED_BY_FINISH_ASC):
    n = len(jobs)
    p = computeP(jobs)

    M = [0] * (n + 1)
    for j in range(1, n + 1):
        weight = jobs[j - 1][1]
        M[j] = max(weight + M[p[j]], M[j - 1])

    chosen = []
    j = n
    while j > 0:
        if jobs[j - 1][1] + M[p[j]] >= M[j - 1]:
            chosen.append(jobs[j - 1][0])
            j = p[j]
        else:
            j -= 1

    return M[n], list(reversed(chosen))


class JobNode:
    def __init__(self, j, name, weight):
        self.j = j            # posición 1..n, 0 = ya no quedan trabajos
        self.name = name
        self.weight = weight
        self.take = None      # hijo izquierdo: OPT(p(j)), se toma el trabajo j
        self.skip = None      # hijo derecho:   OPT(j - 1), se omite el trabajo j
        self.value = 0        # OPT(j)
        self.took = False     # True si tomar el trabajo j es la mejor rama


def buildJobTree(jobs=SORTED_BY_FINISH_ASC, j=None, p=None):
    if p is None:
        p = computeP(jobs)
    if j is None:
        j = len(jobs)
    if j == 0:
        return JobNode(0, "∅", 0)

    name, weight, _, _ = jobs[j - 1]
    node = JobNode(j, name, weight)
    node.take = buildJobTree(jobs, p[j], p)
    node.skip = buildJobTree(jobs, j - 1, p)
    take_value = weight + node.take.value
    node.took = take_value >= node.skip.value
    node.value = max(take_value, node.skip.value)
    return node


def countNodes(node):
    if node is None:
        return 0
    return 1 + countNodes(node.take) + countNodes(node.skip)


def printJobTree(node, prefix="", branch=""):
    if node.j == 0:
        print(f"{prefix}{branch}OPT(0) = 0")
        return
    print(f"{prefix}{branch}OPT({node.j}) {node.name} w={node.weight} = {node.value}")
    child_prefix = prefix + ("│   " if branch.startswith("├") else "    ")
    printJobTree(node.take, child_prefix, f"├─ tomar (+{node.weight}) ")
    printJobTree(node.skip, child_prefix, "└─ omitir ")


TREE_NODE_LIMIT = 80  # con más nodos no se puede leer, los niveles más profundos se cortan


def fitDepth(root, limit=TREE_NODE_LIMIT):
    counts = []  # counts[d] = número de nodos en la profundidad d
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth == len(counts):
            counts.append(0)
        counts[depth] += 1
        if node.j:
            stack.append((node.take, depth + 1))
            stack.append((node.skip, depth + 1))
    shown, depth = 0, -1
    while depth + 1 < len(counts) and shown + counts[depth + 1] <= limit:
        depth += 1
        shown += counts[depth]
    return depth, len(counts) - 1


def plotJobTree(root, max_depth=None, show=True, save_path=None):
    total = countNodes(root)
    fit, full_depth = fitDepth(root)
    if max_depth is None:
        max_depth = fit
    cut = max_depth < full_depth

    # Cuántas veces se calcula cada subproblema OPT(j) en el árbol completo
    seen = {}

    def countCalls(node):
        seen[node.j] = seen.get(node.j, 0) + 1
        if node.j:
            countCalls(node.take)
            countCalls(node.skip)

    countCalls(root)

    # Las hojas reciben posiciones x consecutivas, los padres quedan centrados sobre sus hijos
    positions = {}
    next_x = [0]

    def layout(node, depth):
        if node.j == 0 or depth == max_depth:
            positions[id(node)] = (next_x[0], -depth)
            next_x[0] += 1
            return
        layout(node.take, depth + 1)
        layout(node.skip, depth + 1)
        x = (positions[id(node.take)][0] + positions[id(node.skip)][0]) / 2
        positions[id(node)] = (x, -depth)

    layout(root, 0)

    width = min(30, max(10, next_x[0] * 0.7))
    height = min(14, max(6, (max_depth + 1) * 1.1))
    fig, ax = plt.subplots(figsize=(width, height))
    fig.canvas.manager.set_window_title("Árbol de recursión")
    font = 10 if next_x[0] < 30 else 8

    def draw(node, depth, on_best_path):
        x, y = positions[id(node)]
        truncated = node.j and depth == max_depth
        if node.j and not truncated:
            for child, is_take in ((node.take, True), (node.skip, False)):
                cx, cy = positions[id(child)]
                best = on_best_path and node.took == is_take
                ax.plot([x, cx], [y, cy], color="red" if best else "gray",
                        linewidth=2.5 if best else 1, zorder=1)
                ax.text((x + cx) / 2, (y + cy) / 2, f"+{node.weight}" if is_take else "omitir",
                        fontsize=font - 1, ha="center", va="center", color="dimgray",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
                draw(child, depth + 1, best)

        if node.j == 0:
            label, color = "0", "lightgray"
        else:
            label = f"{node.name}\n{node.value}" + ("\n…" if truncated else "")
            color = "orange" if seen[node.j] > 1 else "lightblue"
        ax.text(x, y, label, ha="center", va="center", fontsize=font, zorder=2,
                bbox=dict(boxstyle="circle", fc=color, ec="red" if on_best_path else "black",
                          linewidth=2 if on_best_path else 1, linestyle="--" if truncated else "-"))

    draw(root, 0, True)

    repeated = sum(count - 1 for j, count in seen.items() if j and count > 1)
    title = f"Árbol de recursión de OPT({root.j}) — {total} llamadas, {repeated} repetidas"
    if cut:
        title += f" (mostrando profundidad {max_depth} de {full_depth}, los nodos punteados continúan)"
    ax.set_title(title + "\nnaranja = subproblema calculado más de una vez, "
                 "camino rojo = decisiones óptimas, +w = tomar trabajo, omitir = omitir trabajo")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()


if __name__ == "__main__":
    jobs = ujobs

    JOB = random.choice(jobs)[0]
    LAST_JOB = jobs[-1][0]
    P_OF_JOB = findPOfJob(LAST_JOB, jobs)
    print(f"trabajo compatible para {LAST_JOB}: {P_OF_JOB}")

    BEST, CHOSEN = weightedSchedule(jobs)
    print(f"peso óptimo {BEST} usando {CHOSEN}")
    plotJobs(JOB, P_OF_JOB, jobs, CHOSEN, show=False)

    # Árbol de recursión de la ecuación de Bellman para los mismos trabajos, también se guarda como imagen
    ROOT = buildJobTree(jobs)
    plotJobTree(ROOT, show=False, save_path="jobs_tree.png")
    plt.show()  # abre ambas ventanas a la vez
