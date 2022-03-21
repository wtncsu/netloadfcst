'''
Classes to produce corrective gradients
'''


import numpy


class Constant_Learning_Rate:
    def __init__(self, learning_rate=1e-3):
        assert learning_rate > 0
        self.epsilon = learning_rate

    def __call__(self, gradient):
        g = numpy.asarray(gradient).reshape(-1)
        return -self.epsilon * g


class RMSProp:
    _delta = 1e-6

    def __init__(self, learning_rate=1e-3, decay_rate=0.01):

        assert learning_rate > 0
        assert 0 <= decay_rate <= 1

        self.epsilon = learning_rate
        self.rho = decay_rate
        self.r = 0

    def __call__(self, gradient):
        g = numpy.asarray(gradient).reshape(-1)
        self.r = self.rho * self.r + (1 - self.rho) * numpy.square(g)
        return -self.epsilon / numpy.sqrt(self._delta + self.r) * g


class Adam:
    _delta = 1e-8

    def __init__(self, step_size=1e-3, decay1=0.9, decay2=0.999):
        assert 0 <= decay1 < 1
        assert 0 <= decay2 < 1
        self.epsilon = step_size
        self.rho1 = decay1
        self.rho2 = decay2
        self.s = 0
        self.r = 0
        self.t = 0

    def __call__(self, gradient):
        g = numpy.asarray(gradient).reshape(-1)
        self.t += 1
        self.s = self.rho1 * self.s + (1 - self.rho1) * g
        self.r = self.rho2 * self.r + (1 - self.rho2) * numpy.square(g)
        shat = self.s / (1 - numpy.power(self.rho1, self.t))
        rhat = self.r / (1 - numpy.power(self.rho2, self.t))

        return -self.epsilon * shat / (numpy.sqrt(rhat) + self._delta)


