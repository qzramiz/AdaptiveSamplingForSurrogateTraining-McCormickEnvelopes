import numpy as np

from envelope_builder import piecewise_envelopes
from interval_builder import IntervalBuilder


class AdaptiveSampler:

    def __init__(self, func, total_pop=50, random_sample=False):
        self.func = func
        self.total_pop = total_pop
        self.random_sample = random_sample


    def get_sample_dist(self, df):
        if self.random_sample:
            idxs, sample_count = np.unique(
                np.random.choice(np.arange(len(df)), self.total_pop, p=(df.gap / df.gap.sum()).values), return_counts=True)
        else:
            sample_count = df.pc * self.total_pop
            idxs = np.arange(len(sample_count))

        sample_count = np.ceil(sample_count).astype(int)
        return idxs, sample_count

    def sample_points(self, domain, points_n, budget_multiplier=2):
        # Settings (reverted original style)
        Xgrid = np.linspace(domain[0], domain[1], points_n * budget_multiplier)
        Ygrid = self.func.f(Xgrid)

        breakpoints = [domain[0], domain[1]]
        # sampled_x = [domain[0], domain[1]]
        sampled_x = []
        BUDGET = points_n

        for _ in range(BUDGET):
            L, U = piecewise_envelopes(self.func, xgrid=Xgrid, ygrid=Ygrid, breakpoints=breakpoints)
            gap = U - L
            idx = int(np.argmax(gap))
            x_new = float(Xgrid[idx])
            # Avoid duplicates (due to grid endpoint alignment)
            if np.any([abs(x_new - b) < 1e-12 for b in breakpoints]):
                for j in np.argsort(-gap):
                    x_candidate = float(Xgrid[int(j)])
                    if np.all([abs(x_candidate - b) > 1e-12 for b in breakpoints]):
                        x_new = x_candidate
                        break
            breakpoints.append(x_new)
            breakpoints.sort()
            sampled_x.append(x_new)

        sampled_x = np.array(sorted(set(sampled_x)))
        sampled_y = self.func.f(sampled_x)

        return sampled_x, sampled_y

    def get_samples(self):
        pass

