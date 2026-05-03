import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

for idx, num_samples in enumerate([100, 500, 1000, 5000]):
    feature_range = range(2, 101)
    ratios = []
    for num_features in feature_range:
        X, _ = make_classification(n_samples=num_samples, n_features=num_features,
                                   n_informative=num_features, n_redundant=0,
                                   random_state=42)
        query_point = X[0].reshape(1, -1)
        X = X[1:]
        distances = np.sqrt(np.sum((X - query_point) ** 2, axis=1))
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)
    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('curse_of_dimensionality.png', dpi=150)
plt.close()