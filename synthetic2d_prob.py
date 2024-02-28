#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ngboost import NGBoost

n_samples = 2400

feature1 = np.linspace(-10, 3, n_samples).reshape(-1, 1)
feature2 = np.linspace(-7, 10, n_samples).reshape(-1, 1)
feature = np.hstack([feature1, feature2])

noise_free1 = np.sin(feature1) + np.cos(feature2)
noise_free2 = np.sin(feature2) * np.cos(feature2)

noise1_std = abs(np.linspace(-0.1, 0.3, n_samples)).reshape(noise_free1.shape)
noise2_std = abs(np.linspace(-0.3, 0.5, n_samples)).reshape(noise_free2.shape)

noise1 = norm(loc=0, scale=noise1_std).rvs()
noise2 = norm(loc=0, scale=noise2_std).rvs()

target1 = noise_free1 + noise1
target2 = noise_free2 + noise2

target = np.hstack([target1, target2])

base_params = {
    'max_n_splits': 8,
    'batch_size':   32,
    'n_epochs':     20
}

model = NGBoost(n_stages=20, base_params=base_params, learn_rate=0.7)

model.fit(feature, target)

predict_mean, predict_std = model.predict(feature)
predict_dist = norm(loc=predict_mean, scale=predict_std)

script_name = Path(__file__).stem


def plot_target(arr):
    index = range(len(arr))
    plt.scatter(index, arr, label='Target', color='white', s=1, alpha=0.7)


outcome_lo = target.min()
outcome_hi = target.max()
resolution = 200

outcomes = np.linspace(outcome_lo, outcome_hi, resolution)

# Add dimension to broadcast against all samples
outcomes = outcomes[..., np.newaxis, np.newaxis]
density = predict_dist.pdf(outcomes)

density1 = density[..., 0]
density2 = density[..., 1]

extent = (0, n_samples, outcome_lo, outcome_hi)

plt.figure()

plt.subplot(211)
plt.imshow(density1, extent=extent, origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Density')
plot_target(target[:, 0])
plt.grid(color='white', alpha=0.2)
plt.legend()

plt.subplot(212)
plt.imshow(density2, extent=extent, origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Density')
plot_target(target[:, 1])
plt.grid(color='white', alpha=0.2)
plt.legend()

plt.show()
