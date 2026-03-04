import numpy as np

from typing import List
from samplers.adaptive_sampler import AdaptiveSampler
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from interval_builder import IntervalBuilder
from envelope_builder import piecewise_envelopes
from samplers.adaptive_sampler import AdaptiveSampler
from config import *


def method0(func, sample_coarse=False, return_domains=False, mccormick_sampling=False):
    sampler = AdaptiveSampler(func, BUDGET, False)
    breakpoints = np.linspace(DOMAIN[0], DOMAIN[1], DOMAIN_SPLITS + 1)

    domains = []
    gaps = []
    for idx in range(len(breakpoints) - 1):
        a, b = breakpoints[idx], breakpoints[idx + 1]
        mc, m = func.f_mccormick(a, b)
        if sample_coarse:
            lb, ub = compute_bounds_on_expr(mc.pyomo_expr)
            gap = [ub - lb]
        else:
            _, _, _, gap = func.envelope_interval(a, b)
        domains.append((a, b))
        gaps.append(max(gap))

    sample_dist = allocate_samples(gaps, BUDGET, MIN_SAMPLES)
    print('Sample Allocations: ', sample_dist)
    sampled_x = []
    for idx in range(len(domains)):
        d1, d2 = domains[idx]
        s = sample_dist[idx]
        if mccormick_sampling:
            samples, _ = sampler.sample_points(domain=domains[idx], points_n=s)
        else:
            if idx > 0:
                samples = np.linspace(d1, d2, s + 1)[1:]
            else:
                samples = np.linspace(d1, d2, s)
        sampled_x.extend(samples)
    sampled_x.sort()
    sampled_x = np.array(sampled_x)
    sampled_y = func.f(sampled_x)

    if return_domains:
        return sampled_x, sampled_y, domains, gaps
    else:
        return sampled_x, sampled_y



def allocate_samples(weights: List[float], total_samples: int, min_per_gap: int) -> List[int]:
    """
    Allocate total_samples across gaps proportionally to weights,
    ensuring at least min_per_gap samples for each gap.

    Args:
        weights (List[float]): List of weights (e.g., gap sizes).
        total_samples (int): Total number of samples to allocate.
        min_per_gap (int): Minimum samples each gap must receive.

    Returns:
        List[int]: Final allocation of samples per gap.
    """
    n = len(weights)
    # Step 1: Assign minimum to each gap
    base_alloc = [min_per_gap] * n
    remaining = total_samples - n * min_per_gap

    if remaining < 0:
        raise ValueError("Not enough samples to satisfy minimum requirement for all gaps.")

    # Step 2: Normalize weights
    total_weight = sum(weights)
    proportions = [w / total_weight for w in weights]

    # Step 3: Proportional allocation
    proportional_alloc = [remaining * p for p in proportions]

    # Step 4: Combine and round
    final_alloc = [base + int(round(prop)) for base, prop in zip(base_alloc, proportional_alloc)]

    # Step 5: Adjust to ensure sum matches total_samples
    diff = total_samples - sum(final_alloc)

    # Distribute remainder (if rounding caused mismatch)
    i = 0
    while diff != 0:
        if diff > 0:
            final_alloc[i % n] += 1
            diff -= 1
        else:
            if final_alloc[i % n] > min_per_gap:
                final_alloc[i % n] -= 1
                diff += 1
        i += 1

    return final_alloc




class Method:
    def __init__(self, func):
        self.func = func

    def _init(self):
        raise NotImplementedError

    def sample(self):
        pass

    def plot(self):
        pass


class Method0(Method):
    def __init__(self, func):
        super().__init__(func)
        self.BUDGET = None
        self.DOMAIN = None
        self.DOMAIN_SPLITS = None
        self.MIN_SAMPLES = None
        self.update_func(func, BUDGET, DOMAIN, DOMAIN_SPLITS, MIN_SAMPLES)

        self.sampled_x = []
        self.breakpoints = []

    def update_func(self, func, budget, domain, splits, min_samples):
        self.func = func
        self.BUDGET = budget
        self.DOMAIN_SPLITS = splits
        self.DOMAIN = domain
        self.MIN_SAMPLES = min_samples

        self._init()

    def _init(self):
        self.sampler = AdaptiveSampler(self.func, self.BUDGET, False)

    def sample(self, sample_coarse=False, return_domains=False, mccormick_sampling=False):
        breakpoints = np.linspace(self.DOMAIN[0], self.DOMAIN[1], self.DOMAIN_SPLITS + 1)
        self.domains = []
        self.gaps = []
        for idx in range(len(breakpoints) - 1):
            a, b = breakpoints[idx], breakpoints[idx + 1]
            mc, m = self.func.f_mccormick(a, b)
            if sample_coarse:
                lb, ub = compute_bounds_on_expr(mc.pyomo_expr)
                gap = [ub - lb]
            else:
                _, _, _, gap = self.func.envelope_interval(a, b)
            self.domains.append((a, b))
            self.gaps.append(max(gap))

        sample_dist = allocate_samples(self.gaps, self.BUDGET, MIN_SAMPLES)
        print('Sample Allocations: ', sample_dist)
        self.sampled_x = []
        for idx in range(len(self.domains)):
            d1, d2 = self.domains[idx]
            s = sample_dist[idx]
            if mccormick_sampling:
                samples, _ = self.sampler.sample_points(domain=self.domains[idx], points_n=s)
            else:
                if idx > 0:
                    samples = np.linspace(d1, d2, s + 1)[1:]
                else:
                    samples = np.linspace(d1, d2, s)
            self.sampled_x.extend(samples)
        self.sampled_x.sort()
        self.sampled_x = np.array(self.sampled_x)
        self.sampled_y = self.func.f(self.sampled_x)

        if return_domains:
            return self.sampled_x, self.sampled_y, self.domains, self.gaps
        else:
            return self.sampled_x, self.sampled_y



    def plot(self):
        pass




