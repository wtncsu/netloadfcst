'''
Nonlinear functions
'''

__all__ = [
    'Sigmoid',
]


import numpy


class Sigmoid:
    def primitive(self, arr):
        arr = numpy.asarray(arr)
        arr = numpy.clip(arr, -500, 500)
        return 1 / (1 + numpy.exp(-arr))

    def derivative(self, arr):
        primitive = self.primitive(arr)
        return primitive * (1 - primitive)


class Lorentzian:
    def primitive(self, arr):
        arr = numpy.asarray(arr)
        return 1 / (1 + numpy.square(arr))

    def derivative(self, arr):
        arr = numpy.asarray(arr)
        return -2 * numpy.sqrt(1 + numpy.square(arr)) * arr
