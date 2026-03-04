import numpy as np

from abc import ABC, abstractmethod
from pyomo.environ import Param, RangeSet
from pyomo.environ import ConcreteModel, Var, sin as pysin, sqrt as pysqrt, exp as pyexp
from pyomo.environ import cos as pycos
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick


def interp_hull(hx, hy, xq: np.ndarray, is_upper: bool = False) -> np.ndarray:
    #     if is_upper:
    #         hull = list(reversed(hull))
    #     hx = np.array([p[0] for p in hull])
    #     hy = np.array([p[1] for p in hull])
    # remove potential duplicate x (keep first)
    if hx.size > 1:
        mask = np.concatenate(([True], np.diff(hx) > 0))
        hx = hx[mask];
        hy = hy[mask]
    return np.interp(xq, hx, hy)


class Function(ABC):
    def __init__(self, envelope_samples=10):
        self.envelope_samples = envelope_samples
        pass

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def f_mccormick(self, a, b):
        pass

    def envelope_interval(self, a, b):
        mc, m = self.f_mccormick(a, b)
        # Sample points to estimate envelope gap
        x_samples = np.linspace(a, b, self.envelope_samples)
        gap = []
        lower = []
        upper = []
        for xi in x_samples:
            m.x.set_value(xi)
            mc.changePoint(m.x, xi)
            upper.append(mc.concave())
            lower.append(mc.convex())

            gap.append(mc.concave() - mc.convex())

        return x_samples, np.array(lower), np.array(upper), np.array(gap)

    def envelope_interval_points(self, a, b, xs):
        def get_upper_lower(m, mc, xi):
            m.x.set_value(xi)
            mc.changePoint(m.x, xi)
            return mc.concave(), mc.convex()

        mc, m = self.f_mccormick(a, b)
        # Sample points to estimate envelope gap
        #     x_samples = np.linspace(a, b, dense)
        x_samples = xs
        upper_lower = np.array([get_upper_lower(m, mc, xi) for xi in x_samples])
        gap = upper_lower[:, 0] - upper_lower[:, 1]

        return x_samples, upper_lower[:, 1], upper_lower[:, 0], gap


class Forrester(Function):

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

    def f(self, x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = (6 * m.x - 2) ** 2 * pysin(12 * m.x - 4)
        mc = McCormick(expr)

        return mc, m


class Schwefel(Function):
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

    def f(self, x):
        return 418.9829 - x * np.sin(np.sqrt(np.abs(x)))

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = 418.9829 - m.x * pysin(pysqrt(abs(m.x)))
        #     expr = (6*m.x - 2)**2 * pysin(12*m.x - 4)
        mc = McCormick(expr)

        return mc, m


class Higdon(Function):
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

    def f(self, x):
        return np.sin((2 * np.pi * x) / 10.) + (0.2 * np.sin((2 * np.pi * x) / 2.5))

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = pysin((2 * np.pi * m.x) / 10.) + (0.2 * pysin((2 * np.pi * m.x) / 2.5))
        #     expr = (6*m.x - 2)**2 * pysin(12*m.x - 4)
        mc = McCormick(expr)

        return mc, m


class GrammacyLee(Function):
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

    def f(self, x):
        return np.sin(10 * np.pi * x) / (2.0 * x) + (x - 1.0) ** 4

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = pysin(10 * np.pi * m.x) / (2.0 * m.x) + (m.x - 1.0) ** 4
        #     expr = (6*m.x - 2)**2 * pysin(12*m.x - 4)
        mc = McCormick(expr)

        return mc, m


class Sigmoid(Function):
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

    def f(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = 1 / (1 + pyexp(-m.x))
        #     expr = (6*m.x - 2)**2 * pysin(12*m.x - 4)
        mc = McCormick(expr)

        return mc, m


class Ackley(Function):
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * np.pi

    def f(self, x):
        term1 = -self.a * np.exp(-self.b * np.sqrt(x ** 2))
        term2 = -np.exp(np.cos(self.c * x))
        ackley_value = term1 + term2 + self.a + np.exp(1)
        return ackley_value

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        term1 = -self.a * pyexp(-self.b * pysqrt(m.x ** 2))
        term2 = -pyexp(pycos(self.c * m.x))
        expr = term1 + term2 + self.a + np.exp(1)
        mc = McCormick(expr)

        return mc, m

class Langermann(Function):
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)
        self.a = np.array([3.0, 5.0, 2.0, 1.0, 7.0])
        self.c = np.array([1.0, 2.0, 5.0, 2.0, 3.0])

    def f(self, x):
        x = np.asarray(x, dtype=float)
        s = (x[..., None] - self.a) ** 2
        terms = self.c * np.exp(-s / np.pi) * np.cos(np.pi * s)
        y = terms.sum(axis=-1)
        return y.item() if np.ndim(y) == 0 else y

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.I = RangeSet(len(self.a))
        m.x = Var(bounds=(a, b))
        m.A = Param(m.I, initialize={i + 1: float(self.a[i]) for i in range(len(self.a))})
        m.C = Param(m.I, initialize={i + 1: float(self.c[i]) for i in range(len(self.c))})

        expr = sum(
            m.C[i] * pyexp(-((m.x - m.A[i]) ** 2) / float(np.pi)) * pycos(float(np.pi) * (m.x - m.A[i]) ** 2)
            for i in m.I
        )
        mc = McCormick(expr)
        return mc, m

class Griewank(Function):

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

    def f(self, x):
        return (x ** 2) / 4000. - np.cos(x) + 1

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        term1 = (m.x ** 2) / 4000.
        term2 = -pycos(m.x) + 1
        expr = term1 + term2
        mc = McCormick(expr)

        return mc, m


class Levy1D(Function):
    """
    w = 1 + (x-1)/4
    f(x) = sin²(πw) + (w-1)²(1 + 10 sin²(πw + 1))
    x ∈ [-10, 10]
    Global min: f(1) = 0
    """

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)
        self.name = "Levy-1D"
        self.default_domain = (-20.0, 20.0)
        self.properties = {
            "multimodal": True,
            "many_local_minima": True,
            "wide_domain": True,
            "difficulty": "hard",
        }

    def f(self, x):
        w = 1.0 + (x - 1.0) / 4.0
        return (np.sin(np.pi * w)) ** 2 + (w - 1.0) ** 2 * (
            1.0 + 10.0 * (np.sin(np.pi * w + 1.0)) ** 2
        )

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        w = 1.0 + (m.x - 1.0) / 4.0
        expr = pysin(np.pi * w) ** 2 + (w - 1.0) ** 2 * (
            1.0 + 10.0 * pysin(np.pi * w + 1.0) ** 2
        )
        mc = McCormick(expr)
        return mc, m

# ═══════════════════════════════════════════════════════════════
# 5. Michalewicz 1D (m=10)
#    Extremely steep, narrow valleys — challenging for relaxation
# ═══════════════════════════════════════════════════════════════

class Michalewicz1D(Function):
    """
    f(x) = -sin(x) * sin^(2m)(x²/π),  m=10
    x ∈ [0, π]
    Near-discontinuous appearance due to m=10 steepness
    """

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)
        self.name = "Michalewicz-1D"
        self.default_domain = (0.0, np.pi)
        self.properties = {
            "steep_valleys": True,
            "near_discontinuous": True,  # m=10 makes valleys razor-thin
            "smooth_but_sharp": True,
            "difficulty": "very hard",
        }

    def __init__(self, m=10, **kwargs):
        super().__init__(**kwargs)
        self.m = m

    def f(self, x):
        return -np.sin(x) * (np.sin(x ** 2 / np.pi)) ** (2 * self.m)

    def f_mccormick(self, a, b):
        m_param = self.m
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = -pysin(m.x) * pysin(m.x ** 2 / np.pi) ** (2 * m_param)
        mc = McCormick(expr)
        return mc, m


# ═══════════════════════════════════════════════════════════════
# 6. Damped Cosine (custom)
#    Exponential decay × oscillation — tests amplitude variation
#    Similar to a 1D slice of the Drop-Wave function
# ═══════════════════════════════════════════════════════════════

class DampedCosine(Function):
    """f(x) = -exp(-0.5x) * cos(5πx),  x ∈ [0, 3]"""

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)
        self.name = "Damped-Cosine"
        self.default_domain = (0.0, 3.0)
        self.properties = {
            "multimodal": True,
            "amplitude_decay": True,     # envelope should tighten with x
            "exp_trig_composition": True, # stresses MC++ composition rules
            "difficulty": "medium",
        }

    def f(self, x):
        return -np.exp(-0.5 * x) * np.cos(5.0 * np.pi * x)

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = -pyexp(-0.5 * m.x) * pycos(5.0 * np.pi * m.x)
        mc = McCormick(expr)
        return mc, m



# ═══════════════════════════════════════════════════════════════
# 9. Sine-Envelope (custom high-frequency stress test)
#    Rapid oscillation that aggressively widens the MC gap
# ═══════════════════════════════════════════════════════════════

class SineEnvelope(Function):
    """
    f(x) = x * sin(20πx) + 2x²
    x ∈ [-1, 1]
    The linear × high-freq-sine product creates wide envelopes.
    """
    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

        self.name = "Sine-Envelope"
        self.default_domain = (-1.0, 1.0)
        self.properties = {
            "multimodal": True,
            "very_high_frequency": True,
            "product_composition": True,  # x * sin(...) is hard for MC++
            "difficulty": "hard",
        }

    def f(self, x):
        return x * np.sin(20.0 * np.pi * x) + 2.0 * x ** 2

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        expr = m.x * pysin(20.0 * np.pi * m.x) + 2.0 * m.x ** 2
        mc = McCormick(expr)
        return mc, m



# ═══════════════════════════════════════════════════════════════
# 12. Bukin-like 1D (adapted from Bukin N.6)
#     Absolute-value ridge with oscillating valley floor
# ═══════════════════════════════════════════════════════════════

class Bukin1D(Function):
    """
    f(x) = 100 * sqrt(|x + 0.01 x²|) + 0.01 |x + 10|
    x ∈ [-15, -3]
    Sharp ridge with a narrow valley
    """

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

        self.name = "Bukin-1D"
        self.default_domain = (-15.0, -3.0)
        self.properties = {
            "non_smooth": True,     # |·| creates kinks
            "narrow_valley": True,
            "sqrt_composition": True,
            "difficulty": "hard",
        }

    def f(self, x):
        return 100.0 * np.sqrt(np.abs(x + 0.01 * x ** 2)) + 0.01 * np.abs(x + 10.0)

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        # sqrt(|·|) ≈ (·^2)^0.25 for MC++ friendliness
        inner = m.x + 0.01 * m.x ** 2
        expr = 100.0 * pysqrt(inner ** 2 + 1e-12) ** 0.5 + 0.01 * pysqrt(
            (m.x + 10.0) ** 2 + 1e-12
        )
        mc = McCormick(expr)
        return mc, m


class Eggholder1D(Function):
    """
    1D Eggholder — derived from the 2D version with x2 fixed at 47
    (near the global minimum region of the 2D function):

    f(x) = -(x + 47) sin(sqrt(|x/2 + 47|)) - x sin(sqrt(|x - 47|))

    x ∈ [-512, 512]

    Preserves the highly irregular, multimodal character of the
    original 2D Eggholder. The sin(sqrt(|·|)) composition creates
    many deep, unpredictable local minima that are extremely
    challenging for both surrogate modeling and convex relaxation.

    Ref: Jamil & Yang (2013), Int. J. Math. Modelling & Num. Opt.
    """

    def __init__(self, envelope_samples=10):
        super().__init__(envelope_samples)

        self.name = "Eggholder-1D"
        self.default_domain = (-512.0, 512.0)
        self.properties = {
            "multimodal": True,
            "many_local_minima": True,
            "highly_irregular": True,
            "abs_sqrt_composition": True,   # |·| + sqrt — hard for MC++
            "wide_domain": True,
            "deceptive": True,              # deep local minima far from global
            "difficulty": "very hard",
        }

    def f(self, x):
        return (
            -(x + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + 47.0)))
            - x * np.sin(np.sqrt(np.abs(x - 47.0)))
        )

    def f_mccormick(self, a, b):
        m = ConcreteModel()
        m.x = Var(bounds=(a, b))
        # Smooth approximation of |·| via sqrt(·² + ε) for MC++
        eps = 1e-8
        abs_term1 = pysqrt((m.x / 2.0 + 47.0) ** 2 + eps)
        abs_term2 = pysqrt((m.x - 47.0) ** 2 + eps)
        expr = (
            -(m.x + 47.0) * pysin(pysqrt(abs_term1))
            - m.x * pysin(pysqrt(abs_term2))
        )
        mc = McCormick(expr)
        return mc, m