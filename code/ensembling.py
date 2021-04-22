# %%

from sklearn.tree import DecisionTreeClassifier
import re
from tools import show, alpha_cmap, rgb_cmap
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
import matplotlib as mpl
# mpl.style.use('classic')
# mpl.style.use('ggplot')
# from cycler import cycler
# mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
# mpl.__version__


#_d = 'def'
_d = 'def2'
#_d = 'uni'
#_d = 'twomoons'


def data_generator(n_samples, seed=42000):

    n_samples //= 2

    def _r(*more): return np.random.randn(*((n_samples,)+more))
    # generate random sample, two components
    np.random.seed(seed)

    if _d == 'def' or _d == 'def2':
        # generate spherical data centered on (20, 20)
        yoff = -3
        G1 = _r(2) + np.array([5, yoff+1])

        # generate zero centered stretched Gaussian data
        C = np.array([[0., -0.7], [3.5, .7]])
        G2 = np.dot(_r(2)/2, C) + np.array([-2, yoff])
        if _d == 'def2':
            def tr(x, n=n_samples//4):
                x[:n] = x[:n] * np.array([[1, -1]]) + np.array([[1, -1]])
            tr(G2)
    elif _d == 'twomoons':
        G1 = _r(2)/[4, 2]
        G1 = np.array([(5 + G1[:, 0]) * np.cos(np.pi + G1[:, 1]), (5 + G1[:, 0]) * np.sin(np.pi + G1[:, 1])]).T
        G1 -= np.array([[-3, 2]])
        G2 = _r(2)/[4, 2]
        G2 = np.array([(5 + G2[:, 0]) * np.cos(G2[:, 1]), (5 + G2[:, 0]) * np.sin(G2[:, 1])]).T
        G2 += np.array([[-3, 2]])
        G1, G2 = G2, G1
    else:
        G1 = G2 = None

    # concatenate the two datnp.random.randn(n_samples, 2)asets into the final training set
    X = np.vstack([G1, G2])
    Y = np.ones(n_samples*2)
    Y[:n_samples] = 0

    return X, Y


X, Y = data_generator(400)
print(X.shape, Y.shape)

# %%


def plot_dataset(X=X, Y=Y, **more):
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='.', **more)
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='x', **more)


def generate_dataset_image(zoomout=.5):
    plot_dataset()
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim([xmin - (xmax-xmin)/2*zoomout, xmax + (xmax-xmin)/2*zoomout])
    plt.ylim([ymin - (ymax-ymin)/2*zoomout, ymax + (ymax-ymin)/2*zoomout])
    show(plt, 'Toy 2D-dataset')


generate_dataset_image()

# %%

LONG, GRID = 5, 30
#LONG, GRID = 20, 50
#LONG, GRID = 40, 100


def uncertainty_maps(gen, n_samples=1000000*LONG, grid_n=GRID):  # 2s
    kwh = dict()  # density=True)
    plot_dataset()
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.clf()
    print("Sampling a lot of points to (re)estimate distributions")
    X, Y = gen(n_samples)
    h = [None, None]
    hall, x2edges, x1edges = np.histogram2d(X[:, 1], X[:, 0], bins=grid_n, range=[[ymin, ymax], [xmin, xmax]], *kwh)
    xx, yy = np.meshgrid(x1edges, x2edges)
    for i in [0, 1]:
        sub = X[Y == i]
        h[i], *_ = np.histogram2d(sub[:, 1], sub[:, 0], bins=(x2edges, x1edges), **kwh)

    def view(hs, cmaps, *args, **kwargs):
        if type(hs) is not list:
            hs = [hs]
            cmaps = [cmaps]
        plot_dataset()
        for h, cmap in zip(hs, cmaps):
            # plt.imshow(h, interpolation='bilinear', cmap=alpha_cmap(cmap), origin='lower', extent=[
            #           x1edges[0], x1edges[-1], x2edges[0], x2edges[-1]])
            plt.pcolormesh(xx, yy, h, cmap=alpha_cmap(cmap))
        show(plt, *args, **dict(ext='png', dpi=300, **kwargs))

    aleatoric = h[0] * h[1]
    aleatoric /= np.max(aleatoric)
    epistemic_abs = np.clip(1 - (hall/(np.max(hall)*.65+0.35*np.median(hall))), 0, 1)
    #data_unc = (aleatoric > 0.005) * (epistemic_abs > 0.95)
    data_unc = 1-np.exp(-aleatoric / 0.005)
    data_unc *= np.exp(-(1 - epistemic_abs) / 0.05)

    blues = rgb_cmap(0, 0, 1)
    view([h[0] ** 0.5, h[1] ** 0.5], [cm.Blues, cm.Oranges], 'Underlying Distributions (contrasted)')
    view(aleatoric**0.25, cm.Greens, 'Aleatoric Uncertainty')
    view(1 - (1-epistemic_abs)**0.25, blues, 'Global Epistemic Uncertainty')
    view([1 - (1-epistemic_abs)**0.25, aleatoric**0.25], [blues, cm.Greens], 'Aleatoric and Epistemic Uncertainties')
    view(data_unc, cm.Purples, '"Aleo-Epistemic" Uncertainty')


uncertainty_maps(data_generator)

# %%


def explore_ensembling(baseclf, poly=None, n_estimators=25, max_samples=0.99999,
                       boundary_alpha=0.3, nicer=False,
                       l_levels=11, l_alpha=0.7,
                       meshgrid_n=301, cmap='coolwarm',  # 'Greys'
                       zoomout=4,
                       seed=42, X=X, Y=Y, more=''):

    clf_name = baseclf.__class__.__name__

    levels = np.linspace(0, 1, l_levels)
    def features(x): return x
    if poly is not None:
        polyfeats = PolynomialFeatures(degree=poly)
        def features(x): return polyfeats.fit_transform(x)
        clf_name += f',p={poly}'

    clf_name += more
    np.random.seed(seed)
    bagging = BaggingClassifier(baseclf, n_estimators=n_estimators, max_samples=max_samples)

    bagging.fit(features(X), Y)

    plot_dataset()
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.clf()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, meshgrid_n),
                         np.linspace(ymin, ymax, meshgrid_n))

    plot_dataset()
    for clf in bagging.estimators_:
        if nicer and hasattr(clf, 'coef_') and hasattr(clf, 'intercept_'):
            coef = clf.coef_
            intercept = clf.intercept_
            c = 0
            def line(x0): return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
            plt.plot([xmin, xmax], [line(xmin), line(xmax)], c='k', ls="-", alpha=boundary_alpha, lw=2)
        else:
            Z = clf.predict(features(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, colors='k', levels=[0.5], alpha=boundary_alpha)

    plt.ylim([ymin, ymax])
    # plt.gca().patch.set_facecolor('#BBB')
    show(plt, f"All classifiers ({clf_name})")

    plot_dataset()
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.clf()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, meshgrid_n),
                         np.linspace(ymin, ymax, meshgrid_n))
    for iclf, clf in enumerate(bagging.estimators_[:3]+[bagging]):

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        try:
            Z = clf.predict_proba(features(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
        except:
            Z = clf.predict(features(np.c_[xx.ravel(), yy.ravel()]))
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap, levels=levels, alpha=l_alpha)
        plt.contour(xx, yy, Z, colors='k', levels=[0.5], linewidths=1, alpha=.5)
        plt.contour(xx, yy, Z, linestyles='dotted', colors='k', levels=[0.5], linewidths=4, alpha=.5)
        plot_dataset()
        show(plt, f"{'Ensemble' if clf.__class__.__name__ == 'BaggingClassifier' else 'Estimator'+str(iclf)} ({clf_name})")
    if zoomout is not None:
        plt.xlim([xmin - (xmax-xmin)/2*zoomout, xmax + (xmax-xmin)/2*zoomout])
        plt.ylim([ymin - (ymax-ymin)/2*zoomout, ymax + (ymax-ymin)/2*zoomout])
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, meshgrid_n),
                             np.linspace(ymin, ymax, meshgrid_n))
        try:
            Z = clf.predict_proba(features(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
        except:
            Z = clf.predict(features(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap, levels=levels, alpha=l_alpha)
        plt.contour(xx, yy, Z, colors='k', levels=[0.5], linewidths=1, alpha=.5)
        plt.contour(xx, yy, Z, linestyles='dotted', colors='k', levels=[0.5], linewidths=4, alpha=.5)
        plot_dataset()
        show(plt, f"{'Ensemble' if clf.__class__.__name__ == 'BaggingClassifier' else 'Estimator'+str(iclf)} (zoomed out) ({clf_name})")


# %%
# simple tests with low sample
explore_ensembling(LogisticRegression(), nicer=True, max_samples=20, more='sub')

# %%
explore_ensembling(LogisticRegression(), max_samples=20, poly=3, more='sub')

# %%
explore_ensembling(SGDClassifier(), nicer=True)
explore_ensembling(SGDClassifier(), poly=3)

# %%
# true bootstrap
explore_ensembling(LogisticRegression(), nicer=True)
explore_ensembling(LogisticRegression(), poly=3)


# %%
# explore_ensembling(MLPClassifier())
#explore_ensembling(MLPClassifier(hidden_layer_sizes=(20, 20, 20)))
explore_ensembling(MLPClassifier(hidden_layer_sizes=(100, 50, 20)))
explore_ensembling(MLPClassifier(hidden_layer_sizes=[10]*10), more='deep')

# %%
explore_ensembling(DecisionTreeClassifier(max_depth=3))

# %%

# custom gamma-1nn-expdecay-logit


def expe_nn_decay(offset, scale, pow=1, zoomout=2,
                  cmap='coolwarm', K=3, force_title=None,
                  show_first=False, show_second=True):

    title = f'$d^{{{pow}}}, K={K}, offset={offset}, scale={scale}$'
    if force_title is not None:
        title = force_title

    def predict_proba(train, trainY, X):
        n = np.shape(X)[0]
        t0 = train[trainY == 0]
        d0 = np.sum((t0[:, None, :] - X[None, :, :]) ** 2, axis=2)
        if K == 1:
            d0 = np.min(d0, axis=0) ** 0.5
        else:
            d0 = np.mean(np.partition(d0, K-1, axis=0)[:K, :]**0.5, axis=0)
        t1 = train[trainY == 1]
        d1 = np.sum((t1[:, None, :] - X[None, :, :]) ** 2, axis=2)
        if K == 1:
            d1 = np.min(d1, axis=0) ** 0.5
        else:
            d1 = np.mean(np.partition(d1, K-1, axis=0)[:K, :]**0.5, axis=0)

        dmin = np.minimum(d0, d1)
        d0 /= dmin
        d1 /= dmin

        res = np.c_[d0, d1]
        res = offset+np.exp(-scale*res**pow)
        res /= np.sum(res, axis=1, keepdims=True)
        return res

    # predict_proba(X, Y, _r(2))

    #levels = np.linspace(0, 1, 12)
    levels = [0, *np.linspace(0.05, .95, 10), 1]
    plot_dataset()
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.clf()
    if show_first:
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 301),
                             np.linspace(ymin, ymax, 301))
        Z = predict_proba(X, Y, np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cmap, levels=levels, alpha=0.7)
        # plt.colorbar()
        plt.contour(xx, yy, Z, linestyles='dotted', colors='k', levels=[0.5], linewidths=4, alpha=.5)
        plot_dataset()
        show(plt, title)

    # second plot
    plt.xlim([xmin - (xmax-xmin)/2*zoomout, xmax + (xmax-xmin)/2*zoomout])
    plt.ylim([ymin - (ymax-ymin)/2*zoomout, ymax + (ymax-ymin)/2*zoomout])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.clf()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 301),
                         np.linspace(ymin, ymax, 301))
    Z = predict_proba(X, Y, np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap, levels=levels, alpha=0.8)
    #plt.contour(xx, yy, Z, colors='g', levels=levels, alpha=1)
    # plt.colorbar()
    plt.contour(xx, yy, Z, colors='k', levels=[0.5], linewidths=1, alpha=.7)
    plt.contour(xx, yy, Z, linestyles='dotted', colors='k', levels=[0.5], linewidths=4, alpha=.5)
    plot_dataset()
    if show_second:
        show(plt, f'{title} (zoomed out)')
    else:
        plt.clf()


expe_nn_decay(0, 1/3, 1, K=5, zoomout=4, force_title='Possible Expected OOD', show_first=True)

# %%


def test(z=.2):
    sc = v = 1 / 3
    #expe_nn_decay(0, sc, 1, K=10, zoomout=z)
    #expe_nn_decay(0, sc, 1, K=3, zoomout=z)
    #expe_nn_decay(0, v, 1, K=1, zoomout=z)
    p_ile = .05  # higher percentile to match
    for pow in [1, 2, 4, .75]:
        x = - np.log(p_ile) / sc
        v = - np.log(p_ile) / x**pow
        #expe_nn_decay(0, v, pow, K=10, zoomout=z)
        #expe_nn_decay(0, v, pow, K=3, zoomout=z)
        expe_nn_decay(0, v, pow, K=1, zoomout=z)


test(4)

# %%
expe_nn_decay(0, .3, 1, show_first=True)
expe_nn_decay(0, .1, 2, show_first=True)

expe_nn_decay(0, .3, 1, show_first=True, K=1)
expe_nn_decay(0, .141, 2, show_first=True, K=1)


# STOP
# %%
for K in [1, 2, 5, 10]:
    expe_nn_decay(0.1,  .1, 2, K=K, zoomout=0)

# %%
for i in np.linspace(0, .3, 10):
    print(i)
    expe_nn_decay(i, 1)

# %%
for i in np.linspace(0.1, 3, 10):
    print(i)
    expe_nn_decay(0.01, i)

# %%
expe_nn_decay(0,    1)
expe_nn_decay(0.01, 1)
expe_nn_decay(0.1, 1)

# %%
expe_nn_decay(0,   .1, 2)
expe_nn_decay(0.1, .1, 2)
expe_nn_decay(0.15, .1, 2)
# -> offset 0 is ok (because /dmin avoid the huge dist, so the all-very small case)
# bigger offset is not good as we loose certainty on the know regions
# squaring can be interesting for faster loss of influence

# NB: probably need something like lof, decay is indep of the other class but related to the distance of neighbors of nearest neighbors

# %%
# %%
for i in np.linspace(0.01, 1, 10):
    print(i)
    expe_nn_decay(0.0, .001, 2)


# old
# %%
for i in [0, 0.1, 1, 10]:
    expe_nn_decay(i, 10)

# %%
for i in [0, 0.1, 1, 10]:
    expe_nn_decay(0.1, i)

# %%
for i in [0, 0.1, 1, 10]:
    expe_nn_decay(0.01, i)

# %%
expe_nn_decay(0.01, 1)
expe_nn_decay(0.01, 1, 2)
#expe_nn_decay(0.01, 1, 3)
#expe_nn_decay(0.01, 10, 2)
expe_nn_decay(0.01, 1, 2)

# %%
expe_nn_decay(0.1, 1, 2)

# %%


def test_partition():
    x = np.random.rand(200)
    plt.plot(x)
    plt.plot(np.partition(x, 50))
    plt.plot(np.partition(x, (10, 20, 30, 40, 50, 60, 70)))
    plt.plot(np.partition(x, (0, 1, 2, 3, 4, 5, 40, 41, 42)))
    plt.show()


test_partition()
# %%
