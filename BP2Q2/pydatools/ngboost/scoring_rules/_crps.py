'''
Continuous Ranked Probability Score
'''
import numpy
import scipy.stats
from functools import singledispatchmethod
from ..distributions import (
    Normal_Parameters,
)


class CRPS:
    '''
    Continuous Ranked Probability Score

    Member function calls are dispatched based on the distribution. The call
    will be redirected to the corresponding CRPS equation for the distribution.
    If an equation is not defined, a NotImplementedError() is thrown.
    '''
    @singledispatchmethod
    @staticmethod
    def primitive(distribution, target):
        raise NotImplementedError()

    @singledispatchmethod
    @staticmethod
    def derivative(distribution, target):
        raise NotImplementedError()

    @singledispatchmethod
    @staticmethod
    def rieman_metric(distribution):
        raise NotImplementedError()

    @staticmethod
    def natural_grad(distribution, target):
        rieman = __class__.rieman_metric(distribution)
        deriv = __class__.derivative(distribution, target)
        return numpy.linalg.solve(rieman, deriv)



@CRPS.primitive.register
def _normal_primitive(dist : Normal_Parameters, target):
    target = target.ravel()
    std = numpy.exp(dist.logstd)
    z = (target - dist.mean) / std

    score = std * (
        z * (2 * scipy.stats.norm.cdf(z) - 1)
        + 2 * scipy.stats.norm.pdf(z)
        - 1 / numpy.sqrt(numpy.pi)
    )
    return score

@CRPS.derivative.register
def _normal_derivative(dist : Normal_Parameters, target):
    target = target.ravel()
    std = numpy.exp(dist.logstd)
    z = (target - dist.mean) / std

    deriv_mean = 1 - 2 * scipy.stats.norm.cdf(z)
    deriv_logstd  = std * (
        2 * scipy.stats.norm.pdf(z) - 1 / numpy.sqrt(numpy.pi))

    return dist.pack(deriv_mean, deriv_logstd)

@CRPS.rieman_metric.register
def _normal_rieman_metric(dist : Normal_Parameters):
    std = numpy.exp(dist.logstd)

    m11 = 1 / std / numpy.sqrt(numpy.pi)
    m22 = std / 2 / numpy.sqrt(numpy.pi)
    m12 = numpy.zeros_like(m11)

    rieman = numpy.asarray([
        [m11, m12],
        [m12, m22]
    ])

    rieman = numpy.moveaxis(rieman, 2, 0)
    return rieman

