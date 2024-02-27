from types import SimpleNamespace

import numpy
from tqdm.auto import trange, tqdm

from .distributions import NormalParameters
from .scoring_rules import CRPS


class MultiOutputModel:
    """
    For fitting parametric distributions with more than one parameters as if
    fitting a single-output model.
    """

    def __init__(self, base_learner, num_outputs):
        self._base_learners = [base_learner.clone() for _ in range(num_outputs)]

    def fit(self, features, targets):
        progress = tqdm(list(zip(self._base_learners, targets)),
                        desc='Parameters', leave=False)

        for model, target in progress:
            model.fit(features, target)

    def predict(self, features):
        targets = [model.predict(features) for model in self._base_learners]
        return targets


def _find_optimal_scaling(Scoring_Rule, Dist_Params, theta_m_prev, f_m_pred,
                          target):
    rho_search_space = numpy.logspace(-10, 2, 100, base=2.0)

    losses = numpy.fromiter(
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
    def __init__(self, base_learner,
                 Dist_Params=NormalParameters,
                 Scoring_Rule=CRPS,
                 n_stages=2000,
                 learn_rate=0.1):
        self._Dist_Params = Dist_Params
        self._Scoring_Rule = Scoring_Rule
        self._n_stages = n_stages
        self._learn_rate = learn_rate

        self._ensemble = [
            MultiOutputModel(base_learner, Dist_Params.num_parameters)
            for _ in range(n_stages)]

        self._scalings = numpy.empty(shape=n_stages)

    def fit(self, features, target):
        n_samples = len(numpy.asarray(features))
        self._theta0 = self._Dist_Params.from_marginal(target).to_array()

        # The initial prediction _theta0 for all target samples is the same and
        # has only one sample, but we need to repeat it by the number of samples
        # so that the base learner does not complain about feature-target length
        # mismatch.
        theta0 = numpy.broadcast_to(
            self._theta0, shape=(n_samples, self._Dist_Params.num_parameters))

        theta_m_prev = theta0.copy()

        progress = trange(self._n_stages, leave=False, desc='Boost stage')

        for boost_stage in progress:
            dist = self._Dist_Params.from_array(theta_m_prev)
            g_m = self._Scoring_Rule.natural_grad(dist, target)

            f_m = self._ensemble[boost_stage]
            f_m.fit(features, targets=self._Dist_Params.unpack(g_m))
            f_m_pred = self._Dist_Params.pack(*f_m.predict(features))

            optimal = _find_optimal_scaling(
                self._Scoring_Rule, self._Dist_Params, theta_m_prev,
                f_m_pred, target)

            self._scalings[boost_stage] = optimal.rho_m
            theta_m_prev -= self._learn_rate * (optimal.rho_m * f_m_pred)

            progress.set_postfix(loss=optimal.loss)

    def predict(self, features):
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
