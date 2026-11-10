"""RoTTA's memory bank: Category-balanced Sampling with Timeliness and Uncertainty (CSTU)."""
import math


class CSTU:
    """Each slot holds [x, uncertainty, age]. Per-class quota = capacity / num_classes.

    A new sample (pseudo-label c) is accepted if
      - class c is below quota and the bank is not full; or
      - class c is below quota, the bank is full, and the worst item of the majority class(es) scores
        higher than the new sample (that item is evicted); or
      - class c is at quota and the worst item of class c scores higher than the new sample.
    Score = lambda_t * sigmoid(age / capacity) + lambda_u * uncertainty / log(num_classes); higher = worse
    (older and/or less certain). Every add() increases the age of all items by 1.
    """

    def __init__(self, capacity, num_classes, lambda_t=1.0, lambda_u=1.0):
        self.capacity, self.num_classes = capacity, num_classes
        self.per_class = capacity / num_classes
        self.lambda_t, self.lambda_u = lambda_t, lambda_u
        self.items = [[] for _ in range(num_classes)]

    def score(self, age, uncertainty):
        return (self.lambda_t / (1 + math.exp(-age / self.capacity))
                + self.lambda_u * uncertainty / math.log(self.num_classes))

    def occupancy(self):
        return sum(len(c) for c in self.items)

    def _evict(self, classes, new_score):
        best = None  # (score, class, index)
        for c in classes:
            for i, (_, unc, age) in enumerate(self.items[c]):
                s = self.score(age, unc)
                if best is None or s >= best[0]:
                    best = (s, c, i)
        if best is None:
            return True
        if best[0] > new_score:
            self.items[best[1]].pop(best[2])
            return True
        return False

    def add(self, x, label, uncertainty):
        new_score = self.score(0, uncertainty)
        if len(self.items[label]) < self.per_class:
            if self.occupancy() < self.capacity:
                accept = True
            else:
                sizes = [len(c) for c in self.items]
                majority = [c for c, n in enumerate(sizes) if n == max(sizes)]
                accept = self._evict(majority, new_score)
        else:
            accept = self._evict([label], new_score)
        if accept:
            self.items[label].append([x, uncertainty, 0])
        for c in self.items:
            for item in c:
                item[2] += 1

    def get(self):
        """Returns (list of x, list of age / capacity)."""
        xs, ages = [], []
        for c in self.items:
            for x, _, age in c:
                xs.append(x)
                ages.append(age / self.capacity)
        return xs, ages
