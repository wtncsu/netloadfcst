import numpy as np
from scipy.stats import norm
from tqdm.auto import trange

from paper_project.probabilistic_fuzzy_tree import (
    ProbabilisticFuzzyTree,
)
from paper_project.probabilistic_fuzzy_tree.membership_functions import (
    SigmoidMF,
)
from paper_project.probabilistic_fuzzy_tree.output_functions import (
    EstimateMean,
)

_SQRT_PI = np.sqrt(np.pi)


def _clipped_log(val):
    val = np.clip(val, a_min=1e-9, a_max=None)
    return np.log(val)


def _clipped_exp(val):
    val = np.clip(val, a_min=-500, a_max=500)
    return np.exp(val)


def _crps_primitive(mu, log_sigma, sample):
    """
    Computes the Continuous Ranked Probability Score of a Gaussian
    distribution against an observed outcome. The inputs must have the same
    shape.
    """
    sigma = _clipped_exp(log_sigma)
    z = (sample - mu) / sigma

    crps = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / _SQRT_PI
    )

    return crps


def _crps_derivative(mu, log_sigma, sample):
    """
    Computes the natural gradients of the CRPS of a Gaussian distribution
    with respect to mean and log standard derivation against an observed
    outcome. The inputs must have the same shape.
    """
    sigma = _clipped_exp(log_sigma)
    z = (sample - mu) / sigma

    ord_mean = 1 - 2 * norm.cdf(z)
    ord_logstd = sigma * (
        2 * norm.pdf(z) - 1 / _SQRT_PI
    )

    nat_mean = ord_mean * sigma * _SQRT_PI
    nat_logstd = ord_logstd * _SQRT_PI * 2 / sigma

    return nat_mean, nat_logstd


class NGBoost:
    def __init__(self, n_stages, learn_rate, base_params):
        self._mu0 = None
        self._log_sigma0 = None

        self._n_stages = n_stages
        self._learn_rate = learn_rate

        self._mu_ensemble = [
            ProbabilisticFuzzyTree(output_func=EstimateMean(),
                                   membership_func=SigmoidMF(),
                                   **base_params)
            for _ in range(n_stages)
        ]

        self._log_sigma_ensemble = [
            ProbabilisticFuzzyTree(output_func=EstimateMean(),
                                   membership_func=SigmoidMF(),
                                   **base_params)
            for _ in range(n_stages)
        ]

        self._scalings = np.empty(n_stages)

    def fit(self, feature, target):
        feature = np.asarray(feature)
        target = np.asarray(target)

        mu0 = feature.mean(axis=0, keepdims=True)
        log_sigma0 = _clipped_log(feature.std(axis=0, keepdims=True))

        self._mu0 = mu0
        self._log_sigma0 = log_sigma0

        mu = mu0
        log_sigma = log_sigma0

        progress = trange(self._n_stages, leave=False, desc='Boost stage')
        for boost_stage in progress:
            mu_model = self._mu_ensemble[boost_stage]
            log_sigma_model = self._log_sigma_ensemble[boost_stage]

            dmu, dlog_sigma = _crps_derivative(mu, log_sigma, target)
            mu_model.fit(feature=feature, target=dmu)
            log_sigma_model.fit(feature=feature, target=dlog_sigma)

            pred_mu = mu_model.predict(feature=feature)
            pred_log_sigma = log_sigma_model.predict(feature=feature)

            scalings = np.logspace(-10, 5, 200, base=2.0)
            scalings = np.expand_dims(scalings, (-1, -2))

            losses = _crps_primitive(
                mu=mu - scalings * pred_mu,
                log_sigma=log_sigma - scalings * pred_log_sigma,
                sample=target
            )
            losses = losses.sum(axis=-1).mean(axis=-1)

            optimal_scaling_index = losses.argmin()
            optimal_scaling = scalings[optimal_scaling_index]

            self._scalings[boost_stage] = optimal_scaling

            mu = mu - self._learn_rate * optimal_scaling * pred_mu
            log_sigma = (
                log_sigma - self._learn_rate * optimal_scaling * pred_log_sigma
            )

    def predict(self, feature):
        mu = self._mu0
        log_sigma = self._log_sigma0

        for boost_stage in range(self._n_stages):
            scaling = self._scalings[boost_stage]

            mu_model = self._mu_ensemble[boost_stage]
            log_sigma_model = self._log_sigma_ensemble[boost_stage]

            mu_pred = mu_model.predict(feature)
            log_sigma_pred = log_sigma_model.predict(feature)

            mu = mu - self._learn_rate * scaling * mu_pred
            log_sigma = log_sigma - self._learn_rate * scaling * log_sigma_pred

        sigma = _clipped_exp(log_sigma)
        return mu, sigma
