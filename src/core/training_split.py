"""Deterministic held-out partitions for training datasets."""
import random


def held_out(items, fraction=0.2, seed=0, minimum=1):
    items = sorted(items)
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(items) < 2 * minimum:
        raise ValueError("need at least %d independent items for disjoint training and validation" % (2 * minimum))
    random.Random(seed).shuffle(items)
    count = max(minimum, min(len(items) - minimum, round(len(items) * fraction)))
    return items[count:], items[:count]
