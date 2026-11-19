"""Three-objective Pareto screening and fixed-bound ideal-point ranking (SI S4)."""

import numpy as np


def rank_candidates(record_ids, predictions, feasible, lower, upper):
    """Return (ID, distance) for feasible members of the original Pareto front.

    Columns follow B50 [T], P1.5/50 [W/kg], UTS [MPa]. Supply the feasibility
    mask from the applicable design and process limits and fixed development
    bounds in physical units. Smaller distance indicates higher priority.
    """
    ids = list(record_ids)
    values = np.asarray(predictions, dtype=float)
    keep_design = np.asarray(feasible)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if len(set(ids)) != len(ids) or values.shape != (len(ids), 3):
        raise ValueError("Supply one unique ID and three predictions per candidate.")
    if keep_design.shape != (len(ids),) or keep_design.dtype != bool:
        raise ValueError("Supply one Boolean feasibility value per candidate.")
    if low.shape != (3,) or high.shape != (3,) or not (high > low).all():
        raise ValueError("Each objective needs fixed, increasing lower and upper bounds.")
    if not (np.isfinite(values).all() and np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("Bounds and predictions must be finite.")

    # Dominance is evaluated on all input candidates before applying feasibility.
    minimize = values * [-1.0, 1.0, -1.0]
    front = []
    for i in np.flatnonzero(keep_design):
        no_worse = (minimize <= minimize[i]).all(axis=1)
        strictly_better = (minimize < minimize[i]).any(axis=1)
        if not (no_worse & strictly_better).any():
            front.append(i)
    normalized = (values[front] - low) / (high - low)
    ideal = np.array([1.0, 0.0, 1.0])
    distances = np.sqrt(np.mean((normalized - ideal) ** 2, axis=1))
    return [(ids[front[j]], float(distances[j])) for j in np.argsort(distances, kind="stable")]
