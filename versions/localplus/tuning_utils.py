"""Small, tested safeguards for validation-selected localplus experiments."""
import numpy as np
import torch.nn.functional as F


def similarity_margin_loss(positive_score, negative_score, margin):
    """Both scores are similarities: a positive must score MORE, not less."""
    return F.relu(float(margin) - positive_score.float() + negative_score.float()).mean()


def complete_training_negatives(candidates, known_answers, n_entities, count):
    """Deduplicate and fill from TRAIN-only nonanswers, never held-out labels."""
    forbidden = set(known_answers)
    selected = list(dict.fromkeys(int(x) for x in candidates if int(x) not in forbidden))[:count]
    if len(selected) < count:
        pool = np.array(sorted(set(range(n_entities)) - forbidden - set(selected)), dtype=np.int64)
        missing = count - len(selected)
        if not len(pool):
            raise ValueError('No distinct training-negative candidates remain')
        selected += np.random.choice(pool, missing, replace=len(pool) < missing).tolist()
    return selected


def prefer_initial(initial_mrr, trained_mrr):
    """Keep the unmodified incoming model on a validation tie or regression."""
    # Lightning's monitor is float32; exact rank statistics are float64.
    return initial_mrr is not None and (trained_mrr is None or initial_mrr >= trained_mrr - 1e-7)
