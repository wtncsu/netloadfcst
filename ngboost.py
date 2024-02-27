from types import SimpleNamespace

import numpy as np
from scipy.stats import norm
from tqdm.auto import trange

from paper_project.probabilistic_fuzzy_tree import (
    ProbabilisticFuzzyTree,
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


def _crps_primitive(mu, sigma, sample):
    """
    Computes the Continuous Ranked Probability Score of a Gaussian
    distribution against an observed outcome. The inputs must have the same
    shape.
    """
    assert mu.shape == sample.shape
    assert sigma.shape == sample.shape

    z = (sample - mu) / sigma

    crps = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / _SQRT_PI
    )

    mean_crps = crps.mean(axis=-1)
    all_targets_total = mean_crps.sum()
    return all_targets_total


def _crps_derivative(mu, sigma, sample):
    """
    Computes the natural gradients of the CRPS of a Gaussian distribution
    with respect to mean and log standard derivation against an observed
    outcome. The inputs must have the same shape.
    """
    assert mu.shape == sample.shape
    assert sigma.shape == sample.shape

    z = (sample - mu) / sigma

    ord_mean = 1 - 2 * norm.cdf(z)
    ord_logstd = sigma * (
        2 * norm.pdf(z) - 1 / _SQRT_PI
    )

    nat_mean = ord_mean * sigma * _SQRT_PI
    nat_logstd = ord_logstd * _SQRT_PI * 2 / sigma

    return nat_mean, nat_logstd


def _find_optimal_scaling(Scoring_Rule, Dist_Params, theta_m_prev, f_m_pred,
                          target):
    rho_search_space = np.logspace(-10, 2, 100, base=2.0)

    losses = np.fromiter(
        (Scoring_Rule.primitive(
            Dist_Params.from_array(theta_m_prev - rho * f_m_pred),
            target).sum()
         for rho in rho_search_space), dtype=float)

    best = losses.argmin()

    optimal = SimpleNamespace()
    optimal.rho_m = rho_search_space[best]
    optimal.loss = losses[best]

    return optimal


class NGBoost:
    __slots__ = (
        '_mu_ensemble',
        '_log_sigma_ensemble',
        '_n_stages',
        '_learn_rate',
        '_scalings',
        '_marginal_mu',
        '_marginal_log_sigma',
    )

    def __init__(self, n_stages, learn_rate, base_params):
        self._n_stages = n_stages
        self._learn_rate = learn_rate

        self._mu_ensemble = [
            ProbabilisticFuzzyTree(output_func=EstimateMean(),
                                   **base_params)
            for _ in range(n_stages)
        ]

        self._log_sigma_ensemble = [
            ProbabilisticFuzzyTree(output_func=EstimateMean(),
                                   **base_params)
            for _ in range(n_stages)
        ]

        self._scalings = np.empty(n_stages)
        self._marginal_mu = None
        self._marginal_log_sigma = None

    def fit(self, feature, target):
        feature = np.asarray(feature)
        target = np.asarray(target)

        mu0 = feature.mean(axis=0, keepdims=True)
        log_sigma0 = _clipped_log(feature.std(axis=0, keepdims=True))

        self._marginal_mu = mu0
        self._marginal_log_sigma = log_sigma0

        prev_mu = mu0
        prev_log_sigma = log_sigma0

        progress = trange(self._n_stages, leave=False, desc='Boost stage')
        for boost_stage in progress:
            prev_sigma = _clipped_exp(prev_log_sigma)

            gradient = _crps_derivative(prev_mu, prev_sigma, target)

            mu_model = self._mu_ensemble[boost_stage]
            log_sigma_model = self._log_sigma_ensemble[boost_stage]

            mu_model.fit(feature=feature, target=prev_mu)
            log_sigma_model.fit(feature=feature, target=prev_log_sigma)

            pred_mu = mu_model.predict(feature=feature)
            pred_log_sigma = log_sigma_model.predict(feature=feature)



            optimal = _find_optimal_scaling(
                self._Scoring_Rule, self._Dist_Params, theta_m_prev,
                f_m_pred, target)

            self._scalings[boost_stage] = optimal.rho_m
            theta_m_prev -= self._learn_rate * (optimal.rho_m * f_m_pred)

            progress.set_postfix(loss=optimal.loss)

    def predict(self, feature):
        feature = np.asarray(feature)

        n_samples = len(numpy.asarray(features))

        # The initial prediction _theta0 for all target samples is the same and
        # has only one sample, but we need to repeat it by the number of samples
        # so that the base learner does not complain about feature-target length
        # mismatch.
        theta0 = numpy.broadcast_to(
            self._theta0, shape=(n_samples, self._Dist_Params.num_parameters))

        sum_f_m_preds = sum(
            rho_m * self._Dist_Params.pack(*f_m.predict(features))
            for f_m, rho_m in zip(self._ensemble, self._scalings)
        )

        theta = theta0 - self._learn_rate * sum_f_m_preds
        return self._Dist_Params.from_array(theta).to_distribution()
