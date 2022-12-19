'''
Decision tree common routines
'''
import numpy
from tqdm.auto import tqdm, trange
from ..containers.multiway_trees import Multiway_Tree, Multiway_Tree_Node



def _impurity(pred, target):
    return numpy.square(target - pred).sum()

def _predict(target):
    return target.mean()



def _get_split_thresholds(feature_val):
    unique_val = numpy.unique(feature_val)

    # midpoints between unique values
    if len(unique_val) > 1:
        return (unique_val[:-1] + unique_val[1:]) / 2

    else:
        return numpy.empty(shape=0)


def _find_best_split(node, min_split_size, min_impurity_drop):

    # Skips repeated calculations if we have already done it before.
    if hasattr(node, 'impurity_drop'):
        return

    node.impurity_drop = None

    features = node.features
    target = node.target
    impurity_before_split = node.impurity
    best_impurity_drop = min_impurity_drop

    min_split_index = min_split_size
    max_split_index = len(features) - min_split_size

    columns = tqdm(features.transpose(), desc='Columns', leave=False)

    for split_col, feature_val in enumerate(columns):

        thresholds = tqdm(_get_split_thresholds(feature_val),
                          desc='Thresholds', leave=False)

        sort_indices = feature_val.argsort()
        sort_features = features[sort_indices]
        sort_target = target[sort_indices]
        sort_feature_val = sort_features[:, split_col]

        for split_threshold in thresholds:

            split_index = numpy.searchsorted(sort_feature_val, split_threshold,
                                             side='right')

            if not (min_split_index <= split_index <= max_split_index):
                continue

            left_target = sort_target[:split_index]
            right_target = sort_target[split_index:]

            left_pred = _predict(left_target)
            right_pred = _predict(right_target)

            left_impurity = _impurity(left_pred, left_target)
            right_impurity = _impurity(right_pred, right_target)

            impurity_after_split = left_impurity + right_impurity
            impurity_drop = impurity_before_split - impurity_after_split

            if impurity_drop > best_impurity_drop:
                best_impurity_drop = impurity_drop

                node.impurity_drop = best_impurity_drop
                node.split_col = split_col
                node.split_threshold = split_threshold

                left = Multiway_Tree_Node()
                right = Multiway_Tree_Node()

                node.left = left
                node.right = right

                left.features = sort_features[:split_index]
                right.features = sort_features[split_index:]

                left.target = left_target
                right.target = right_target

                left.impurity = left_impurity
                right.impurity = right_impurity

                left.pred = left_pred
                right.pred = right_pred



def build_tree(features, target,
               min_impurity_drop, max_num_splits, min_split_size):

    tree = Multiway_Tree()
    root_node = Multiway_Tree_Node()
    tree.add_node(root_node, parent=None)

    root_node.features = features
    root_node.target = target
    root_node.pred = _predict(target)
    root_node.impurity = _impurity(root_node.pred, target)

    progress = trange(max_num_splits, leave=False, desc='Splits')

    for _ in progress:
        for leaf in tree.leaves:
            _find_best_split(leaf, min_split_size, min_impurity_drop)

        search_nodes = [node for node in tree.leaves
                        if node.impurity_drop is not None]

        if not search_nodes:
            break

        impurity_drop = numpy.fromiter(
            (node.impurity_drop for node in search_nodes), dtype=float)

        best = impurity_drop.argmax()
        best_node = search_nodes[best]

        tree.add_node(best_node.left, parent=best_node, direction='left')
        tree.add_node(best_node.right, parent=best_node,direction='right')

        # cleanup temporary variables
        del best_node.left, best_node.right
        del best_node.impurity_drop
        del best_node.features, best_node.target

    return tree
