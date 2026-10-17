"""Pyro implementation of Attentive State-Space Models
evaluation utils and functions."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix


def align_from_means(learned_means, true_means):

    learned      = np.asarray(learned_means)
    truth        = np.asarray(true_means)
    costs        = ((learned[:, None] - truth[None]) ** 2).sum(-1)
    row, col     = linear_sum_assignment(costs)
    mapping      = np.empty(learned.shape[0], dtype=int)
    mapping[row] = col

    return mapping

def recovery_metrics(probs, truth, mask, mapping):
    
    probs, truth, mask    = np.asarray(probs), np.asarray(truth), np.asarray(mask, dtype=bool)
    aligned               = np.empty_like(probs)
    aligned[..., mapping] = probs
    p, z                  = aligned[mask], truth[mask]
    pred                  = p.argmax(-1)

    return {
        "accuracy": float(np.mean(pred == z)),
        "adjusted_rand_index": float(adjusted_rand_score(z, pred)),
        "state_nll": float(-np.log(np.clip(p[np.arange(len(z)), z], 1e-10, 1)).mean()),
        "confusion_matrix": confusion_matrix(z, pred, labels=np.arange(probs.shape[-1])).tolist(),
        "n_visits": len(z),
    }
