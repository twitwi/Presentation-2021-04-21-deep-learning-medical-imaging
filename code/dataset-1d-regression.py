
# %%

from tools import show
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

# %%

np.random.seed(42)
n_samples = 300

X = np.random.uniform(0, 1, n_samples)
Y = X*0 + 1
p = np.s_[:n_samples//3]
X[p] += 3.5
X[p] = np.min(X[p]) + (X[p] - np.min(X[p]))*1.5
Y[p] = np.sin((X[p] - np.min(X[p])) / 1.5) * 2 + 2
Y[p] += np.random.uniform(-1, 1, X[p].shape)*.8
#Y[p] += np.random.normal(0, .1, X[p].shape)
X[p] += np.random.normal(0, .5, X[p].shape)
p = np.s_[n_samples // 3:2*n_samples//3]
X[p] += 7.5
X[p] = np.min(X[p]) + (X[p] - np.min(X[p]))*2
Y[p] = np.sin(-np.pi*1.3/2 + (X[p] - np.min(X[p]))*1.5) * .5 + 5
Y[p] += np.random.normal(0, .05, X[p].shape)
p = np.s_[2*n_samples // 3:-n_samples//6]
X[p] += 10
X[p] = np.min(X[p]) + (X[p] - np.min(X[p]))/2
Y[p] = np.sin((X[p] - np.min(X[p]))*20)/1.5 + 5.5
p = np.s_[-n_samples//6:]
X[p] += 10.5
X[p] = np.min(X[p]) + (X[p] - np.min(X[p]))*2
Y[p] = np.sin(-np.pi*1.3/2 + (X[p] - np.min(X[p]))*1.5) * -.5 + 5
Y[p] += np.random.normal(0, .05, X[p].shape)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim([0, 15])
plt.ylim([0, 10])
plt.scatter(X, Y, marker='.', c='darkred', s=10)

show(plt, 'Toy 1D-dataset')
# %%

# TODO COULD DO SOME BAGGING HERE ALSO
