'''
Tree Data Structure
'''

from collections import deque


class Binary_Tree_Node:
    __slots__ = (
        '__dict__',
        '_parent',
        '_l_child',
        '_r_child',
        '_is_on_left',
    )

    def __init__(self):
        self._parent = None
        self._l_child = None
        self._r_child = None
        self._is_on_left = None

    @property
    def parent(self):
        return self._parent

    @property
    def left_child(self):
        return self._l_child

    @property
    def right_child(self):
        return self._r_child

    @property
    def ancestors(self):
        p = self._parent
        while p is not None:
            yield p
            p = p.parent

    @property
    def is_on_left(self):
        return self._is_on_left

    @property
    def is_leaf(self):
        return (self._l_child is None) and (self._r_child is None)

    @property
    def is_root(self):
        return self._parent is None

    @property
    def descendants(self):
        '''
        Returns descendants in level-order traversal root, left, right, ...
        '''
        nodes_to_search = deque()
        nodes_to_search.append(self)

        while nodes_to_search:
            node = nodes_to_search.popleft()

            if node._l_child is not None:
                nodes_to_search.append(node._l_child)
                yield node._l_child

            if node._r_child is not None:
                nodes_to_search.append(node._r_child)
                yield node._r_child



class Binary_Tree:
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
    def nodes(self):
        return self._nodes

    @property
    def root(self):
        return self._root

    @property
    def leaves(self):
        yield from self._leaves

    def __contains__(self, node):
        return node in self._nodes

    def add_node(self, node, parent=None, left_side=True):
        '''
        Add new node to tree
        :param node Node: node to add
        :param left_side bool: True if node is on left side of parent
        :param parent Node: parent of node, None for root node
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

            if left_side:
                assert parent._l_child is None, 'existing l_child'
                parent._l_child = node
                node._is_on_left = True
            else:
                assert parent._r_child is None, 'existing r_child'
                parent._r_child = node
                node._is_on_left = False

        self._nodes.append(node)
        self._leaves.append(node)
        node._l_child = None
        node._r_child = None
        node._parent = parent


    def topological_ordering(self):
        if self._root is not None:
            yield self._root
            yield from self._root.descendants

