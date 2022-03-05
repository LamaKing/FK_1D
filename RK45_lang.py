from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step
import numpy as np

class RK45_lang(RK45):
    """ Solve RK with fixed time step. Needed when using random number generators

    Use nskip to do internally some steps instead of keep calling the function.
    """

    def set_nskip(self, nskip):
        self.ncalls = 0
        self.nskip = nskip

    def set_step(self, h):
        self.h = h
        self.h_abs = np.abs(h)

    def _step_impl(self):
        t0 = self.t
        y0 = self.y

        h_abs = self.h_abs
        h = self.h
        nskip = self.nskip

        step_accepted = False
        step_rejected = False

        t = t0
        y = y0
        f = self.f
        for ii in range(nskip):
            y, f = rk_step(self.fun, t, y, f, h, self.A,
                           self.B, self.C, self.K)
            #y, f = rk_step(self.fun, t, y, f, h)
            t += h
            self.ncalls += 1

        step_accepted = True # Cannot fail, is fix step.

        self.h_previous = h
        self.y_old = y0

        self.t = t
        self.y = y

        self.h_abs = h_abs
        self.f = f

        return True, None
