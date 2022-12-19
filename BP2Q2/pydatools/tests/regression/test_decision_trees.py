'''
Unit tests for decision trees
'''

import unittest
import numpy
from pydatools.regression import (
    Decision_Tree_Regressor,
)



class Test_Decicion_Tree(unittest.TestCase):

    def test_decision_tree_default(self):
        values = numpy.linspace(0, 10, 10)
        features = values.reshape(-1, 1)
        target = values

        model = Decision_Tree_Regressor()
        model.fit(features, target)
        pred = model.predict(features)

        numpy.testing.assert_almost_equal(pred, target)
