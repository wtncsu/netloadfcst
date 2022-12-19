'''
Decision tree for regression
'''
import numpy
from ._abstract import Regressor
from ._decision_trees_common import build_tree



def _forward_prop(tree, features):
    pred = numpy.empty(shape=len(features))

    tree.root.fire_stren = True

    for node in tree.topological_ordering():

        if node.is_leaf:
            pred[node.fire_stren] = node.pred

        else:
            feature_val = features[:, node.split_col]
            degree_truth = (feature_val <= node.split_threshold)
            degree_false = ~degree_truth

            node.child('left').fire_stren = node.fire_stren & degree_truth
            node.child('right').fire_stren = node.fire_stren & degree_false

        del node.fire_stren

    return pred



class Decision_Tree_Regressor(Regressor):
    def __init__(self, **kwargs):
        self._param_dict = {
            'min_split_size': 1,
            'min_impurity_drop': 0.0,
            'max_num_splits': 1000,
        }

        self._param_dict.update(kwargs)
        self._tree = None

    def clone(self):
        '''
        Returns a cloned decision tree with the same parameters but not the
        fitted tree itself.
        '''
        return Decision_Tree_Regressor(**self._param_dict)

    def fit(self, features, target):
        '''
        Fit features and output, resulting in a crisp tree
        :param features ndarray: array of shape (n_samples, n_features,)
        :param output ndarray: array of shape (n_samples,)
        '''
        features = numpy.atleast_2d(features)
        target = numpy.asarray(target).ravel()
        self._tree = build_tree(features, target, **self._param_dict)

    def predict(self, features):
        '''
        Predict output based on features
        :param features ndarray: array of shape (n_samples, n_features,)
        :returns: array of shape (n_samples, )
        '''
        features = numpy.atleast_2d(features)
        return _forward_prop(self._tree, features)
