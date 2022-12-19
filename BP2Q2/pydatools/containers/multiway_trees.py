'''
Multiway Tree Data Structure
'''

from collections import (
    deque,
)

class Multiway_Tree_Node:
    __slots__ = (
        '__dict__',
        '_parent',
        '_direction',
        '_children',
    )

    def __init__(self):
        self._parent = None
        self._direction = None
        self._children = dict()

    def __copy__(self):
        '''
        Copies only non-topology attributes
        '''
        from copy import copy

        other = Multiway_Tree_Node()
        other.__dict__ = copy(self.__dict__)
        return other

    def __deepcopy__(self, memo):
        '''
        Copies only non-topology attributes
        '''
        from copy import deepcopy

        other = Multiway_Tree_Node()
        other.__dict__ = deepcopy(self.__dict__, memo)
        return other

    @property
    def parent(self):
        return self._parent

    def child(self, direction):
        if direction in self._children.keys():
            return self._children[direction]
        else:
            return None

    @property
    def direction(self):
        return self._direction

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return not self._children

    @property
    def ancestors(self):
        node = self._parent

        while node is not None:
            yield node
            node = node._parent

    @property
    def descendants(self):
        '''
        Returns descendants in level-order traversal root, left, right, ...
        '''
        nodes_to_search = deque()
        nodes_to_search.append(self)

        while nodes_to_search:
            node = nodes_to_search.popleft()
            for child in node._children.values():
                nodes_to_search.append(child)
                yield child



class Multiway_Tree:
    __slots__ = (
        '__dict__',
        '_root',
        '_leaves',
        '_nodes',
    )

    def __init__(self):
        self._root = None
        self._leaves = list()
        self._nodes = list()

    @property
    def root(self):
        return self._root

    @property
    def nodes(self):
        return self._nodes

    @property
    def leaves(self):
        return self._leaves

    def __contains__(self, node):
        return node in self._nodes

    def topological_ordering(self):
        if self._root is not None:
            yield self._root
            yield from self._root.descendants

    def __deepcopy__(self, memo):
        from copy import deepcopy

        new_tree = Multiway_Tree()
        new_tree.__dict__ = deepcopy(self.__dict__, memo)

        new_nodes = {
            id(node): deepcopy(node, memo)
            for node in self.nodes
        }

        for original_node in self.topological_ordering():
            new_node = new_nodes[id(original_node)]

            new_node_parent = (
                new_nodes[id(original_node.parent)]
                    if original_node.parent is not None
                else None
            )

            new_node_direction = original_node.direction

            new_tree.add_node(new_node, parent=new_node_parent,
                              direction=new_node_direction)

        return new_tree

    def add_node(self, node, parent=None, direction='left'):
        '''
        Add node to tree
        '''
        assert node not in self._nodes, 'node already in tree'

        if self._root is None:
            assert parent is None, 'root node shall not have parent'
            self._root = node

        else:
            assert parent is not None, 'missing parent'
            assert parent in self._nodes, 'unrecognized parent'

            if parent in self._leaves:
                self._leaves.remove(parent)

            assert direction not in parent._children.keys(), 'child exists'

            parent._children[direction] = node
            node._direction = direction

        self._nodes.append(node)
        self._leaves.append(node)
        node._parent = parent

