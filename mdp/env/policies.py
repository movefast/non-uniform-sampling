from typing import Any, Callable, Sequence

import numpy as np


class Policy:
    def __init__(self, probs: Callable[[Any], Sequence[float]]):
        self.probs = probs

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return np.random.choice(len(action_probabilities), p=action_probabilities)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: Sequence[Sequence[float]]):
    return Policy(lambda s: probs[s])

def fromActionArray(probs: Sequence[float]):
    return Policy(lambda s: probs)
