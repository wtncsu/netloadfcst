'''
Unit Tests for NGBoost Parametric Distributions
'''
import unittest
import numpy
from pydatools.ngboost.distributions import (
    Normal_Parameters,
)


class Test_Normal_Distribution(unittest.TestCase):

    def test_pack_unpack(self):
        mean = numpy.asarray([1.0, 2.0, 3.0])
        logstd = numpy.asarray([0.0, 1.0, 2.0])

        to_array = Normal_Parameters.pack(mean, logstd)
        self.assertEqual(to_array.shape, (3, 2))

        dist = Normal_Parameters.from_array(to_array)

        numpy.testing.assert_almost_equal(dist.mean, mean)
        numpy.testing.assert_almost_equal(dist.logstd, logstd)


    def test_from_marginal(self):
        target = numpy.asarray([[1, 2], [3, 4]])

        dist = Normal_Parameters.from_marginal(target)

        numpy.testing.assert_almost_equal(dist.mean, 2.5)
        numpy.testing.assert_almost_equal(dist.logstd, 0.11157177565710492)

