from lisa.pelt import *

LOAD_AVG_PERIOD = 32
window = 1024 * 1024 * 1e-9
LOAD_AVG_MAX = 47741 * (window / 1024)


def decay_load(val, n):
    y = (1 / 2)**(1 / 32)
    return val * (y ** n)


class sched_avg:
    def __init__(self, init=0):
        self.period_contrib = 0
        self.util_avg = init
        self.util_sum = self.util_avg * LOAD_AVG_MAX / 1024


def update_load_sum(sa, running, true_delta):
    # delta_us = true_delta / 1024 * 1e9
    delta = true_delta
    sa.period_contrib += delta
    periods = sa.period_contrib // window

    if periods:
        window_fraction = sa.period_contrib % window

        # C1
        sa.util_sum += running * (window - (sa.period_contrib - delta))
        sa.util_sum = decay_load(sa.util_sum, periods)

        # c2 = LOAD_AVG_MAX - decay_load(LOAD_AVG_MAX, periods) - window
        y = (1 / 2)**(1 / 32)
        c2 = sum(
            running * window * (y ** n)
            for n in range(1, int(periods))
        )

        sa.util_sum += c2

        # C3
        sa.util_sum += running * window_fraction

        divider = (LOAD_AVG_MAX - window + window_fraction)
        sa.util_avg = 1024 * sa.util_sum / divider
        sa.period_contrib = window_fraction
    else:
        sa.util_sum += running * delta

    return sa.util_avg
