'''
Unit tests for regression metrics
'''


import unittest

from datools.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    sum_of_squared_error,
    mean_absolute_percent_error,
    mean_absolute_percent_full_scale_error,
    mean_bias_error,
    coefficient_of_determination,
)


class Test_All(unittest.TestCase):

    def test_case1(self):
        test_actual = [1.0, 2.0, 3.0, 4.0]
        test_pred   = [0.0, 0.0, 0.0, 0.0]

        test_keys = {
            mean_absolute_error : 2.5,
            mean_squared_error : 7.5,
            root_mean_squared_error : 2.73861278753,
            sum_of_squared_error : 30,
            mean_absolute_percent_error : 100,
            mean_absolute_percent_full_scale_error: 62.5,
            mean_bias_error : 2.5,
            coefficient_of_determination: 0,
        }

        for test_func, test_key in test_keys.items():
            self.assertAlmostEqual(test_key, test_func(test_pred, test_actual))
