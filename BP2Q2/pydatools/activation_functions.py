'''
Activation functions
'''
import numpy



class Sigmoid:
    @staticmethod
    def primitive(arr):
        arr = numpy.clip(arr, -500, 500)
        return 1 / (1 + numpy.exp(-arr))

    @staticmethod
    def derivative(arr):
        primitive = __class__.primitive(arr)
        return primitive * (1 - primitive)
