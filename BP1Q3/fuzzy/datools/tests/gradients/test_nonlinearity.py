'''
Unit tests for nonlinear functions
'''


import unittest

from datools.gradients.nonlinearity import Sigmoid


class Test_Sigmoid(unittest.TestCase):
    def test_primitive(self):
        sigmoid = Sigmoid()

        test_inputs = [-1, 0, 1]
        test_keys   = [0.26894142137, 0.5, 0.73105857863]

        test_outputs = sigmoid.primitive(test_inputs)

        for test_key, test_output in zip(test_keys, test_outputs):
            self.assertAlmostEqual(test_key, test_output)

    def test_derivative(self):
        sigmoid = Sigmoid()

        test_inputs = [-1, 0, 1]
        test_keys   = [0.19661193324, 0.25, 0.19661193324]

        test_outputs = sigmoid.derivative(test_inputs)

        for test_key, test_output in zip(test_keys, test_outputs):
            self.assertAlmostEqual(test_key, test_output)
