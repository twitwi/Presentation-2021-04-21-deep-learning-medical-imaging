
# %%

from tools import show
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# %%

simple = [
    (5, multivariate_normal(mean=[-1, 2], cov=[4, 6])),  # [[4, -3.5], [2.5, 5]])),
    (2, multivariate_normal(mean=[2, -2], cov=[4, 2])),
]
normals = [
    (1, multivariate_normal(mean=[3, 0], cov=1.1)),
    (2, multivariate_normal(mean=[-1, -4], cov=2)),
    (5, multivariate_normal(mean=[-1, 2], cov=[[4, -3.5], [2.5, 5]])),
    (1, multivariate_normal(mean=[4, -5], cov=[2, .5])),
]


def landscape(x, y, normals=normals):
    xy = np.transpose([x, y], (1, 2, 0))
    return -np.sum([p*n.pdf(xy) for p, n in normals], axis=0)


xx, yy = np.meshgrid(np.linspace(-7, 7, 101),
                     np.linspace(-7, 7, 103))


def plot_landscape(ti, *args, xx=xx, yy=yy):
    zz = landscape(xx, yy, *args)
    m = np.min(zz)
    M = np.max(zz)
    plt.plot(xx, zz)
    plt.show()
    plt.contourf(xx, yy, zz, cmap='bwr', levels=np.linspace(m, M, 1 + 4), alpha=.3)
    plt.colorbar(ticks=[])
    plt.contour(xx, yy, m+(M-zz), cmap='Greys', levels=np.linspace(m, M, 1+8), linewidths=1.25)
    show(plt, ti)


plot_landscape('Loss landscape')
plot_landscape('Loss landscape (simpler model)', simple)
# %%


# %%
