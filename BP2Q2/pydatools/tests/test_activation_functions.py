'''
Unit tests for activation functions
'''
import numpy
import unittest
from pydatools.activation_functions import Sigmoid


class Test_Sigmoid(unittest.TestCase):
    def test_primitive(self):
        test_inputs = numpy.asarray([-1, 0, 1])
        test_keys   = numpy.asarray([0.26894142137, 0.5, 0.73105857863])

        test_outputs = Sigmoid.primitive(test_inputs)

        numpy.testing.assert_almost_equal(test_outputs, test_keys)

    def test_derivative(self):
        test_inputs = [-1, 0, 1]
        test_keys   = [0.19661193324, 0.25, 0.19661193324]

        test_outputs = Sigmoid.derivative(test_inputs)

        numpy.testing.assert_almost_equal(test_outputs, test_keys)
