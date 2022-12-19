'''
Fuzzy Decision Trees
'''
import numpy
from tqdm.auto import trange
from copy import deepcopy
from ._abstract import Regressor
from ._decision_trees_common import build_tree
from ..activation_functions import Sigmoid
from ..gradient_descent.optimizers import Adam



def _tune_loss_func(pred, target):
    return numpy.square(target - pred).mean()



def _add_feature_val(tree, features):
    for node in tree.nodes:
        if not node.is_leaf:
            node.feature_val = features[:, node.split_col]

def _cleanup_feature_val(tree):
    for node in tree.nodes:
        if not node.is_leaf:
            del node.feature_val



def _forward_prop(tree):
    pred = numpy.zeros(shape=len(tree.root.feature_val))

    param_pred = tree.param_pred
    param_threshold = tree.param_threshold
    param_gain = tree.param_gain

    tree.root.fire_stren = 1.0

    for node in tree.topological_ordering():
        param_index = node.param_index

        if node.is_leaf:
            pred += param_pred[..., param_index] * node.fire_stren

        else:
            node.activation = (
                -param_gain[..., param_index] *
                (node.feature_val - param_threshold[..., param_index])
            )

            left_child = node.child('left')
            right_child = node.child('right')

            left_child.degree_truth = Sigmoid.primitive(node.activation)
            right_child.degree_truth = 1.0 - left_child.degree_truth

            left_child.fire_stren = node.fire_stren * left_child.degree_truth
            right_child.fire_stren = node.fire_stren * right_child.degree_truth

    return pred

def _cleanup_forward_prop(tree):
    for node in tree.nodes:
        if node.is_root:
            del node.activation

        elif node.is_leaf:
            del node.degree_truth

        else:
            del node.activation
            del node.degree_truth

        del node.fire_stren



def _backward_prop(tree, dl_dyhat):

    param_pred = tree.param_pred
    param_threshold = tree.param_threshold
    param_gain = tree.param_gain

    param_grad_pred = tree.param_grad_pred
    param_grad_threshold = tree.param_grad_threshold
    param_grad_gain = tree.param_grad_gain

    for node in reversed(list(tree.topological_ordering())):
        param_index = node.param_index

        if node.is_leaf:
            dyhat_dybar = node.fire_stren
            dyhat_dr = param_pred[..., param_index]

            node.grad_fire_stren = dl_dyhat * dyhat_dr
            param_grad_pred[..., param_index] = dl_dyhat * dyhat_dybar

        else:
            dl_dri_left = node.child('left').grad_fire_stren
            dl_dri_right = node.child('right').grad_fire_stren

            dri_dmup_left = node.fire_stren
            dri_dmup_right = -node.fire_stren

            dl_dmup = (
                dl_dri_left * dri_dmup_left +
                dl_dri_right * dri_dmup_right
            )

            dl_dmu = dl_dmup
            dmu_da = Sigmoid.derivative(node.activation)
            dl_da = dl_dmu * dmu_da

            da_dg = param_threshold[..., param_index] - node.feature_val
            da_dt = param_gain[..., param_index]

            param_grad_gain[..., param_index] = dl_da * da_dg
            param_grad_threshold[..., param_index] = dl_da * da_dt

            if not node.is_root:
                dri_drp_left = node.degree_truth
                dri_drp_right = 1 - node.degree_truth

                dl_drp = (
                    dl_dri_left * dri_drp_left +
                    dl_dri_right * dri_drp_right
                )

                node.grad_fire_stren = dl_drp



def _cleanup_backward_prop(tree):
    for node in tree.nodes:
        if not node.is_root:
            del node.grad_fire_stren


def _make_initial_gain(tree):
    for node in tree.nodes:
        if not node.is_leaf:
            node.gain = 0.0


def _make_parameters(tree, batch_size):
    leaves = tree.leaves
    non_leaves = [node for node in tree.nodes if not node.is_leaf]

    n_leaves = len(leaves)
    n_non_leaves = len(non_leaves)

    # Collect all tunable parameters into numpy array for performance.
    param_pred = numpy.empty((1, n_leaves))
    param_threshold = numpy.empty((1, n_non_leaves))
    param_gain = numpy.empty((1, n_non_leaves))

    param_grad_pred = numpy.empty((batch_size, n_leaves))
    param_grad_threshold = numpy.empty((batch_size, n_non_leaves))
    param_grad_gain = numpy.empty((batch_size, n_non_leaves))

    tree.param_pred = param_pred
    tree.param_threshold = param_threshold
    tree.param_gain = param_gain

    tree.param_grad_pred = param_grad_pred
    tree.param_grad_threshold = param_grad_threshold
    tree.param_grad_gain = param_grad_gain

    for param_index, node in enumerate(leaves):
        node.param_index = param_index
        param_pred[..., param_index] = node.pred
        del node.pred

    for param_index, node in enumerate(non_leaves):
        node.param_index = param_index
        param_threshold[..., param_index] = node.split_threshold
        param_gain[..., param_index] = node.gain

        del node.split_threshold
        del node.gain


def _cleanup_collect_parameters(tree):
    del tree.param_grad_pred, tree.param_grad_threshold, tree.param_grad_gain



def _tune_tree(tree, features, target, pred_optimizer, threshold_optimizer,
               gain_optimizer, n_epochs, batch_size):

    _make_initial_gain(tree)
    _make_parameters(tree, batch_size)

    n_samples = len(features)
    batch_ranges = range(batch_size, n_samples, batch_size)
    n_batches = len(batch_ranges)
    losses = numpy.empty((n_epochs, n_batches))
    epoch_progress = trange(n_epochs, desc='Tune epoch', leave=False)

    random = numpy.random.default_rng()

    pred_optimizer.reset()
    threshold_optimizer.reset()
    gain_optimizer.reset()

    pred_optimizer.set_parameters(tree.param_pred)
    threshold_optimizer.set_parameters(tree.param_threshold)
    gain_optimizer.set_parameters(tree.param_gain)

    for epoch in epoch_progress:
        # randomly shuffle all samples at the beginning of each epoch
        shuffle = random.permutation(range(n_samples))

        # separate into minibatches
        features_split = numpy.array_split(features[shuffle], batch_ranges)
        target_split = numpy.array_split(target[shuffle], batch_ranges)

        batch_progress = trange(n_batches, desc='Tune minibatch', leave=False)

        epoch_losses = losses[epoch]

        for batch in batch_progress:
            batch_features = features_split[batch]
            batch_target = target_split[batch]

            _add_feature_val(tree, batch_features)

            batch_target_pred = _forward_prop(tree)

            loss = _tune_loss_func(batch_target_pred, batch_target)
            grad = -2 * (batch_target - batch_target_pred)

            epoch_losses[batch] = loss

            _backward_prop(tree, grad)

            pred_optimizer(tree.param_grad_pred.mean(axis=0))
            gain_optimizer(tree.param_grad_gain.mean(axis=0))
            threshold_optimizer(tree.param_grad_threshold.mean(axis=0))

        epoch_progress.set_postfix(min=epoch_losses.min(),
                                   max=epoch_losses.max(),
                                   avg=epoch_losses.mean())

    # cleanup temporary variables
    _cleanup_feature_val(tree)
    _cleanup_collect_parameters(tree)
    _cleanup_forward_prop(tree)
    _cleanup_backward_prop(tree)



def _update_default_params(default, kwargs):
    for key, val in kwargs.items():
        if key in default:
            default[key] = val

def _make_tune_param_dict(kwargs):
    params = {
        'pred_optimizer': Adam(),
        'threshold_optimizer': Adam(),
        'gain_optimizer': Adam(),
        'batch_size': 128,
        'n_epochs': 500,
    }
    _update_default_params(params, kwargs)
    return params

def _make_build_param_dict(kwargs):
    params = {
        'min_split_size': 1,
        'min_impurity_drop': 0.0,
        'max_num_splits': 1000,
    }
    _update_default_params(params, kwargs)
    return params



class Fuzzy_Decision_Tree_Regressor(Regressor):
    def __init__(self, **kwargs):
        self._build_param_dict = _make_build_param_dict(kwargs)
        self._tune_param_dict = _make_tune_param_dict(kwargs)
        self._tree = None

    def from_decision_tree(self, other):
        assert other._tree is not None

        new_model = Fuzzy_Decision_Tree_Regressor()
        new_model._build_param_dict = None
        new_model._tune_param_dict = {**self._tune_param_dict}
        new_model._tree = deepcopy(other._tree)

        return new_model

    def clone(self):
        '''
        Returns a cloned instance with the same parameters but not the fitted
        tree itself.
        '''
        other = Fuzzy_Decision_Tree_Regressor()
        other._build_param_dict = deepcopy(self._build_param_dict)
        other._tune_param_dict = self._tune_param_dict
        return other

    def fit(self, features, target):
        '''
        Fit features and output, resulting in a crisp tree
        :param features ndarray: array of shape (n_samples, n_features,)
        :param output ndarray: array of shape (n_samples,)
        '''
        features = numpy.atleast_2d(features)
        target = numpy.asarray(target).ravel()

        if self._tree is None:
            self._tree = build_tree(features, target, **self._build_param_dict)

        _tune_tree(self._tree, features, target, **self._tune_param_dict)

    def predict(self, features):
        '''
        Predict output based on features
        :param features ndarray: array of shape (n_samples, n_features,)
        :returns: array of shape (n_samples, )
        '''
        features = numpy.atleast_2d(features)

        _add_feature_val(self._tree, features)

        pred = _forward_prop(self._tree)

        _cleanup_forward_prop(self._tree)
        _cleanup_feature_val(self._tree)

        return pred
