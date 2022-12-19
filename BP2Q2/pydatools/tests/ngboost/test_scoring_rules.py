'''
Unit Tests for NGBoost scoring rules
'''
import unittest
import numpy
from pydatools.ngboost.distributions import (
    Normal_Parameters,
)

from pydatools.ngboost.scoring_rules import (
    CRPS,
)


class Test_Normal_CRPS(unittest.TestCase):

    def test1(self):
        mean = numpy.asarray([
            13910.8976544744000,
            13947.3945733993000,
            14161.5030398579000,
            15380.0658520247000,
            11535.3061615156000,
        ])

        std = numpy.asarray([
            15462.2651126510000,
            15740.7576922208000,
            15871.1217833955000,
            16938.8023346310000,
            13140.0113464823000,
        ])

        target = numpy.asarray([
            3887.1100000000000,
            7455.6000000000000,
            37952.0000000000000,
            9119.8400000000000,
            23506.0000000000000,
        ])

        primitive_key = numpy.asarray([
            6118.7327930019700,
            4731.7552294776400,
            15768.6035401179000,
            4871.1607615168800,
            7143.7303972550600,
        ])

        ord_grad_mean_key = numpy.asarray([
            0.4831923865859,
            0.3199679878789,
            -0.8661212256323,
            0.2883038093473,
            -0.6377103805244,
        ])

        ord_grad_logstd_key = numpy.asarray([
            1275.3149136064000,
            2654.5887821037100,
            -4836.8508454027100,
            3066.3138010038100,
            -490.1053256259890,
        ])

        nat_grad_mean_key = numpy.asarray([
            13242.4436743841000,
            8927.0321773733000,
            -24364.7097572317000,
            8655.8160257870000,
            -14852.3153922266000,
        ])

        nat_grad_logstd_key = numpy.asarray([
            0.2923810726657,
            0.5978284147955,
            -1.0803388726005,
            0.6417100332486,
            -0.1322204446935,
        ])

        logstd = numpy.log(std)
        dist = Normal_Parameters(mean, logstd)
        primitive = CRPS.primitive(dist, target)
        ord_grad = CRPS.derivative(dist, target)
        nat_grad = CRPS.natural_grad(dist, target)
        ord_grad_mean, ord_grad_logstd = dist.unpack(ord_grad)
        nat_grad_mean, nat_grad_logstd = dist.unpack(nat_grad)

        numpy.testing.assert_almost_equal(primitive, primitive_key)
        numpy.testing.assert_almost_equal(ord_grad_mean, ord_grad_mean_key)
        numpy.testing.assert_almost_equal(ord_grad_logstd, ord_grad_logstd_key)
        numpy.testing.assert_almost_equal(nat_grad_mean, nat_grad_mean_key)
        numpy.testing.assert_almost_equal(nat_grad_logstd, nat_grad_logstd_key)


