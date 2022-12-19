'''
NGBoost Parametric Distributions
'''
import numpy
import scipy.stats



class Normal_Parameters:
    __slots__ = (
        'mean',
        'logstd',
    )

    num_parameters = 2

    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd

    @staticmethod
    def from_array(parameters):
        mean, logstd = __class__.unpack(parameters)

        return Normal_Parameters(mean, logstd)

    @staticmethod
    def from_marginal(target):
        mean = target.mean()
        logstd = numpy.log(target.std())

        return Normal_Parameters(mean, logstd)

    @staticmethod
    def pack(mean, logstd):
        '''
        Combines to_array (or gradients) into a single numpy array
        '''
        mean = mean.ravel()
        logstd = logstd.ravel()
        return numpy.stack((mean, logstd), axis=-1)

    @staticmethod
    def unpack(parameters):
        mean = parameters[..., 0]
        logstd = parameters[..., 1]
        return mean, logstd

    def to_array(self):
        return self.pack(self.mean, self.logstd)

    def to_distribution(self):
        std = numpy.exp(self.logstd)
        return scipy.stats.norm(loc=self.mean, scale=std)
