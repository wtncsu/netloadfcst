'''
Unit tests for multiway trees
'''

import unittest
from pydatools.containers.multiway_trees import (
    Multiway_Tree,
    Multiway_Tree_Node,
)

class Test_Multiway_Tree_Node(unittest.TestCase):
    def test_shallowcopy(self):
        from copy import copy

        original = Multiway_Tree_Node()
        test_obj = object()
        original.test_obj = test_obj

        copied = copy(original)

        self.assertIsNot(copied, original)
        self.assertIs(copied.test_obj, test_obj)

    def test_deepcopy(self):
        from copy import deepcopy

        original = Multiway_Tree_Node()
        test_obj = object()

        original.test_obj = test_obj

        # testing robustness against recursion
        original.node = original

        copied = deepcopy(original)

        self.assertIsNot(copied, original)
        self.assertIsNot(copied.test_obj, test_obj)
        self.assertIsNot(copied.node, original)
        self.assertIsNot(copied.node.test_obj, test_obj)



class Test_Tree(unittest.TestCase):

    def test_shallowcopy(self):
        from copy import copy

        tree = Multiway_Tree()
        node = Multiway_Tree_Node()
        test_obj = object()

        tree.test_obj = test_obj
        tree.add_node(node)
        tree_copy = copy(tree)

        self.assertIsNot(tree_copy, tree)
        self.assertIs(tree_copy.root, tree.root)
        self.assertIs(tree_copy.test_obj, test_obj)

    def test_deepcopy(self):
        from copy import deepcopy

        original = Multiway_Tree()
        node = Multiway_Tree_Node()
        test_obj = object()

        original.test_obj = test_obj

        # testing robustness against recursion
        original.tree = original

        original.add_node(node)
        copied = deepcopy(original)

        self.assertIsNot(copied, original)
        self.assertIsNot(copied.root, node)
        self.assertIsNot(copied.test_obj, test_obj)
        self.assertIsNot(copied.tree, original)
        self.assertIsNot(copied.tree.root, node)
        self.assertIsNot(copied.tree.test_obj, test_obj)

    def test_grandchild(self):
        tree = Multiway_Tree()

        root = Multiway_Tree_Node()
        l_child = Multiway_Tree_Node()
        r_child = Multiway_Tree_Node()
        l_grandchild = Multiway_Tree_Node()

        tree.add_node(root)
        tree.add_node(l_child, parent=root, direction='left')
        tree.add_node(r_child, parent=root, direction='right')
        tree.add_node(l_grandchild, parent=l_child, direction='left')

        self.assertIn(root, tree)
        self.assertIn(l_child, tree)
        self.assertIn(r_child, tree)
        self.assertIn(l_grandchild, tree)

        self.assertNotIn(root, tree.leaves)
        self.assertNotIn(l_child, tree.leaves)

        self.assertIn(r_child, tree.leaves)
        self.assertIn(l_grandchild, tree.leaves)

        self.assertEqual(tree.root, root)

        self.assertEqual(root.parent, None)
        self.assertEqual(root.child('left'), l_child)
        self.assertEqual(root.child('right'), r_child)

        self.assertEqual(l_child.parent, root)
        self.assertEqual(l_child.child('left'), l_grandchild)
        self.assertEqual(l_child.child('right'), None)

        self.assertEqual(r_child.parent, root)
        self.assertEqual(r_child.child('left'), None)
        self.assertEqual(r_child.child('right'), None)

        self.assertEqual(l_grandchild.parent, l_child)
        self.assertEqual(l_grandchild.child('left'), None)
        self.assertEqual(l_grandchild.child('right'), None)

        self.assertEqual(l_child.direction, 'left')
        self.assertEqual(r_child.direction, 'right')
        self.assertEqual(l_grandchild.direction, 'left')

        self.assertFalse(root.is_leaf)
        self.assertFalse(l_child.is_leaf)
        self.assertTrue(r_child.is_leaf)
        self.assertTrue(l_grandchild.is_leaf)

        self.assertTrue(root.is_root)
        self.assertFalse(l_child.is_root)
        self.assertFalse(r_child.is_root)
        self.assertFalse(l_grandchild.is_root)

    def test_ancestors_descendants(self):
        tree = Multiway_Tree()

        root = Multiway_Tree_Node()
        l_child = Multiway_Tree_Node()
        r_child = Multiway_Tree_Node()
        l_grandchild = Multiway_Tree_Node()

        tree.add_node(root)
        tree.add_node(l_child, parent=root, direction='left')
        tree.add_node(r_child, parent=root, direction='right')
        tree.add_node(l_grandchild, parent=l_child, direction='left')

        self.assertEqual(list(root.ancestors), [])
        self.assertEqual(list(l_child.ancestors), [root])
        self.assertEqual(list(r_child.ancestors), [root])
        self.assertEqual(list(l_grandchild.ancestors), [l_child, root])

        self.assertEqual(list(root.descendants),
                         [l_child, r_child, l_grandchild])
        self.assertEqual(list(l_child.descendants), [l_grandchild])
        self.assertEqual(list(r_child.descendants), [])
        self.assertEqual(list(l_grandchild.descendants), [])

    def test_traverse_empty(self):
        # Traversals of empty tree should not raise any exceptions
        tree = Multiway_Tree()
        for _ in tree.topological_ordering():
            pass

    def test_topological_ordering(self):
        tree = Multiway_Tree()

        root = Multiway_Tree_Node()
        l_child = Multiway_Tree_Node()
        r_child = Multiway_Tree_Node()
        l_grandchild = Multiway_Tree_Node()

        tree.add_node(root)
        tree.add_node(l_child, parent=root, direction='left')
        tree.add_node(r_child, parent=root, direction='right')
        tree.add_node(l_grandchild, parent=l_child, direction='left')

        test_key = [root, l_child, r_child, l_grandchild]

        self.assertEqual(list(tree.topological_ordering()), test_key)

    def test_death_existing_node(self):
        tree = Multiway_Tree()

        root = Multiway_Tree_Node()
        l_child = Multiway_Tree_Node()
        r_child = Multiway_Tree_Node()
        l_grandchild = Multiway_Tree_Node()

        tree.add_node(root)
        tree.add_node(l_child, parent=root, direction='left')
        tree.add_node(r_child, parent=root, direction='right')
        tree.add_node(l_grandchild, parent=l_child, direction='left')

        with self.assertRaises(AssertionError):
            tree.add_node(l_grandchild)

    def test_death_existing_root(self):
        tree = Multiway_Tree()

        root = Multiway_Tree_Node()
        another_root = Multiway_Tree_Node()

        tree.add_node(root)

        with self.assertRaises(AssertionError):
            tree.add_node(another_root, parent=None)

    def test_death_existing_l_child(self):
        tree = Multiway_Tree()

        root = Multiway_Tree_Node()
        l_child = Multiway_Tree_Node()
        another_l_child = Multiway_Tree_Node()

        tree.add_node(root)
        tree.add_node(l_child, parent=root, direction='left')

        with self.assertRaises(AssertionError):
            tree.add_node(another_l_child, parent=root, direction='left')
