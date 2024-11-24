import numpy as np


def jensen_shannon_distance(p: np.array, q: np.array, epsilon: float = 1e-10) -> float:
    """Compute the Jensen-Shannon distance between two probability distributions."""
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    m = 0.5 * (p_norm + q_norm)

    js_divergence = 0.5 * np.sum(
        p_norm * np.log(p_norm / (m + epsilon) + epsilon)
    ) + 0.5 * np.sum(q_norm * np.log(q_norm / (m + epsilon) + epsilon))
    js_distance = np.sqrt(max(0, js_divergence))
    return float(js_distance)
