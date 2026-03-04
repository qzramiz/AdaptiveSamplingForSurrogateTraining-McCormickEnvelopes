import numpy as np



def piecewise_envelopes(func, xgrid, ygrid, breakpoints):
    bps = sorted(set([float(max(xgrid.min(), min(xgrid.max(), b))) for b in breakpoints]))
    L = np.zeros_like(ygrid);
    U = np.zeros_like(ygrid)
    for i in range(len(bps) - 1):
        # Get bounds
        a, b = bps[i], bps[i + 1]
        # filter the points indices
        m = (xgrid >= a) & (xgrid <= b)
        # get the points
        xi = xgrid[m]
        yi = ygrid[m]

        # find the lower and over estimators
        _, lo, up, g = func.envelope_interval_points(a, b, xi)

        # replace the values
        L[m] = interp_hull(xi, lo, xi, is_upper=False)
        U[m] = interp_hull(xi, up, xi, is_upper=True)
    return L, U

def interp_hull(hx, hy, xq, is_upper = False):
    #     if is_upper:
    #         hull = list(reversed(hull))
    #     hx = np.array([p[0] for p in hull])
    #     hy = np.array([p[1] for p in hull])
    # remove potential duplicate x (keep first)
    if hx.size > 1:
        mask = np.concatenate(([True], np.diff(hx) > 0))
        hx = hx[mask]
        hy = hy[mask]
    return np.interp(xq, hx, hy)