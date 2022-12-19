'''
Regression Metrics
'''



def mean_absolute_error(pred, actual):
    pred = pred.ravel()
    actual = actual.ravel()

    error = actual - pred

    return abs(error).mean()

def mean_squared_error(pred, actual):
    from numpy import square
    pred = pred.ravel()
    actual = actual.ravel()

    error = actual - pred

    return square(error).mean()


def root_mean_squared_error(pred, actual):
    from numpy import sqrt
    return sqrt(mean_squared_error(pred, actual))


def sum_of_squared_error(pred, actual):
    from numpy import square
    pred = pred.ravel()
    actual =  actual.ravel()

    error = actual - pred

    return square(error).sum()


def mean_absolute_percent_error(pred, actual):
    pred = pred.ravel()
    actual = actual.ravel()

    error = actual - pred
    error_percent = error / actual * 100

    return abs(error_percent).mean()


def mean_absolute_percent_full_scale_error(pred, actual, fullscale=None):
    pred = pred.ravel()
    actual = actual.ravel()

    if fullscale is None:
        fullscale = abs(actual).max()

    error = actual - pred
    error_percent = error / fullscale * 100

    return abs(error_percent).mean()


def weighted_mean_absolute_percent_error(pred, actual):
    pred = pred.ravel()
    actual = actual.ravel()

    return abs(actual - pred).sum() / abs(actual).sum()

def mean_bias_error(pred, actual):
    pred = pred.ravel()
    actual = actual.ravel()

    error = actual - pred

    return error.mean()

def coefficient_of_determination(pred, actual):
    from numpy import square
    pred = pred.ravel()
    actual = actual.ravel()

    error = actual - pred

    residual_sum_of_square = square(error).sum()
    total_sum_of_square = square(actual).sum()

    return 1 - residual_sum_of_square / total_sum_of_square


def mean_pinball_loss(pred, actual, quantile=0.5):
    from numpy import where

    loss = where(actual >= pred,
                 (actual - pred) * quantile,
                 (pred - actual) * (1 - quantile))

    return loss.mean()


