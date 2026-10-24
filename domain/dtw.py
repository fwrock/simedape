import math
import numpy as np
from dtaidistance import dtw
from joblib import Memory

_memory = Memory("./cachedir", verbose=0)


# ---------------------------------------------------------------------------
# DTW matrix computation (custom implementation preserved from original)
# ---------------------------------------------------------------------------

def warping_paths(s1, s2, window=None, max_dist=None, max_step=None,
                  max_length_diff=None, penalty=None, psi=None):
    """Computes the full DTW matrix between two series.
    Returns (distance, dtw_matrix)."""
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return np.inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0

    dtw_matrix = np.full((r + 1, c + 1), np.inf)
    for i in range(psi + 1):
        dtw_matrix[0, i] = 0
        dtw_matrix[i, 0] = 0

    last_under_max_dist = 0
    i0 = 1
    i1 = 0
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        for j in range(max(0, i - max(0, r - c) - window + 1),
                       min(c, i + max(0, c - r) + window)):
            d = (s1[i] - s2[j]) ** 2
            if max_step is not None and d > max_step:
                continue
            dtw_matrix[i1, j + 1] = d + min(
                dtw_matrix[i0, j],
                dtw_matrix[i0, j + 1] + penalty,
                dtw_matrix[i1, j] + penalty,
            )
            if max_dist is not None:
                if dtw_matrix[i1, j + 1] <= max_dist:
                    last_under_max_dist = j
                else:
                    dtw_matrix[i1, j + 1] = np.inf
                    if prev_last_under_max_dist < j + 1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            return np.inf, dtw_matrix

    dtw_matrix = np.sqrt(dtw_matrix)
    if psi == 0:
        d = dtw_matrix[i1, min(c, c + window - 1)]
    else:
        ir = i1
        ic = min(c, c + window - 1)
        vr = dtw_matrix[ir - psi:ir + 1, ic]
        vc = dtw_matrix[ir, ic - psi:ic + 1]
        mir = np.argmin(vr)
        mic = np.argmin(vc)
        if vr[mir] < vc[mic]:
            dtw_matrix[ir - psi + mir + 1:ir + 1, ic] = -1
            d = vr[mir]
        else:
            dtw_matrix[ir, ic - psi + mic + 1:ic + 1] = -1
            d = vc[mic]
    return d, dtw_matrix


def best_path(paths, x1, x2):
    """Traceback do caminho ótimo na matriz DTW."""
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    p = []
    if paths[i, j] != -1:
        fun = 1 if x1[i - 1] > x2[j - 1] else 0
        p.append((i - 1, j - 1, paths[i, j], fun))
    while i > 0 and j > 0:
        c = np.argmin([paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]])
        if c == 0:
            i, j = i - 1, j - 1
        elif c == 1:
            i -= 1
        else:
            j -= 1
        if paths[i, j] != -1:
            fun = 1 if x1[i - 1] > x2[j - 1] else 0
            p.append((i - 1, j - 1, paths[i, j], fun))
    p.pop()
    p.reverse()
    return p


best_path_cached = _memory.cache(best_path)


def get_warping_path_fast(ts_1, ts_2):
    """Calcula o caminho de warping DTW usando dtaidistance (C backend)."""
    d, paths = dtw.warping_paths_fast(
        ts_1, ts_2,
        window=math.floor(len(ts_2) * 0.10),
        psi=2,
    )
    bp = best_path(paths, ts_1, ts_2)
    return bp, d


# ---------------------------------------------------------------------------
# Time translation: reconstrói série a partir do warping path
# ---------------------------------------------------------------------------

def time_translation(path, s):
    """Reconstrói a série estimada usando o caminho de warping e a série do
    ponto de simulação."""
    serie = np.zeros(len(s))
    t = path[0]
    if t[3]:
        serie[t[1]] = s[t[0]] - t[2]
    else:
        serie[t[1]] = s[t[0]] + t[2]
    for pos in range(1, len(path)):
        t = path[pos]
        before = path[pos - 1]
        if t[3]:
            serie[t[1]] = s[t[0]] - math.sqrt(max(t[2] ** 2 - before[2] ** 2, 0))
        else:
            serie[t[1]] = math.sqrt(max(t[2] ** 2 - before[2] ** 2, 0)) + s[t[0]]
    return serie


def estimate(time_series, warping_path, std, mean):
    """Estima uma série temporal: aplica time translation e denormaliza."""
    translated = time_translation(warping_path, time_series)
    return translated * std + mean


def pruning(series):
    """Remove valores negativos substituindo por 0."""
    return [max(0, v) for v in series]
