import numpy as np
from scipy.special import factorial

def MCMT(sl, st, fn):
    # Minimum controled maximum thresholds
    # sl: smoothing length (in frame)
    # st: smoothing increment (in frame)
    # fn: frame number

    # Erlang Distribution
    intval = 0.01
    x_values = np.arange(0.00, 3.01, intval)
    cd = np.zeros(len(x_values))
    it = 0

    for x in sl * x_values:
        it += 1
        value = 1
        itr = 1
        for n in range(1, sl):
            value *= x / n
            itr += value
        cd[it-1] = 1 - np.exp(-x) * itr

    G = sl * x_values
    GC = cd

    # Minimum controled maximum thresholds
    bl = fn * st * (1 + np.log(sl / st)) / sl
    pl = np.log(G+0.00000001) * (sl - 1) - G - np.log(factorial(sl - 1)+0.00000001) + np.log(1 - GC+0.00000001) * (bl - 1)
    pu = np.log(G+0.00000001) * (sl - 1) - G - np.log(factorial(sl - 1)+0.00000001) + np.log(GC+0.00000001) * (bl - 1)

    # Check for invalid values in pl and pu
    pl[np.isnan(pl) | np.isinf(pl)] = -np.inf
    pu[np.isnan(pu) | np.isinf(pu)] = -np.inf

    el = np.sum(np.exp(pl) * G) / np.sum(np.exp(pl))
    # eu = np.sum(np.exp(pu) * G) / np.sum(np.exp(pu))
    Fu = np.cumsum(np.exp(pu)) / np.sum(np.exp(pu))

    # Check if Fu contains valid values
    if np.all(Fu <= 0.9):
        th_s = np.nan
    else:
        th_s = G[np.where(Fu > 0.9)[0][0]] / el

    if np.all(Fu <= 0.5):
        th_n = np.nan
    else:
        th_n = G[np.where(Fu > 0.5)[0][0]] / el

    return th_s, th_n

# Example usage:
# th_s, th_n = MCMT(sl, st, fn)