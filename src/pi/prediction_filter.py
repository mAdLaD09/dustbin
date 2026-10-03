from collections import deque, Counter
from dataclasses import dataclass

@dataclass
class StablePrediction:
    label: str
    confidence: float

class PredictionFilter:
    def __init__(self, window_size=7, minimum_votes=5, threshold=.8):
        self.window = deque(maxlen=window_size); self.minimum_votes=minimum_votes; self.threshold=threshold
    def reset(self): self.window.clear()
    def update(self, label: str, confidence: float) -> StablePrediction | None:
        self.window.append((label, float(confidence)))
        if len(self.window) < self.minimum_votes: return None
        counts=Counter(x[0] for x in self.window); winner,votes=counts.most_common(1)[0]; avg=sum(c for l,c in self.window if l==winner)/votes
        return StablePrediction(winner,avg) if votes >= self.minimum_votes and avg >= self.threshold else None
