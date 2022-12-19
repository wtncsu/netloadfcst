'''
Gradient descent optimizers
'''
import numpy



class Constant_Learning_Rate_Optimizer:
    def __init__(self, learn_rate=1e-3):
        assert learn_rate > 0.0
        self._epsilon = learn_rate
        self._parameters = None

    def set_parameters(self, parameters):
        self._parameters = parameters

    def __call__(self, grad):
        self._parameters -= self._epsilon * grad

    def reset(self):
        self._parameters = None



class RMSProp:
    _delta = 1e-6

    def __init__(self, learn_rate=1e-3, decay=0.01):
        assert learn_rate > 0.0
        assert 0.0 <= decay <= 1.0

        self._epsilon = learn_rate
        self._rho = decay
        self._r = 0.0
        self._parameters = None

    def set_parameters(self, parameters):
        self._parameters = parameters

    def __call__(self, grad):
        self._r = self._rho * self._r + (1 - self._rho) * numpy.square(grad)
        self._parameters -= (
            self._epsilon / numpy.sqrt(self._delta + self._r) * grad)

    def reset(self):
        self._parameters = None
        self._r = 0.0



class Adam:
    _delta = 1e-8

    def __init__(self, step_size=1e-2, decay1=0.9, decay2=0.999):
        assert step_size > 0.0
        assert 0.0 <= decay1 < 1.0
        assert 0.0 <= decay2 < 1.0

        self._parameters = None

        self._epsilon = step_size
        self._rho1 = decay1
        self._rho2 = decay2
        self._s = 0.0
        self._r = 0.0
        self._t = 0

    def set_parameters(self, parameters):
        self._parameters = parameters

    def __call__(self, grad):
        self._t += 1
        self._s = self._rho1 * self._s + (1 - self._rho1) * grad
        self._r = self._rho2 * self._r + (1 - self._rho2) * numpy.square(grad)

        shat = self._s / (1 - numpy.power(self._rho1, self._t))
        rhat = self._r / (1 - numpy.power(self._rho2, self._t))

        self._parameters -= (
            self._epsilon * shat / (numpy.sqrt(rhat) + self._delta))

    def reset(self):
        self._parameters = None
        self._s = 0.0
        self._r = 0.0
        self._t = 0
