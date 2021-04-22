
import re
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

import os
NOSHOW = 'BATCH' in os.environ
#NOSHOW = True

def show(plt, name=None, fname=None, hide_title=False, ext='svg', dpi=None):
    if name is not None:
        if not hide_title:
            plt.title(name)
        if fname is None:
            fname = name.lower()
            fname = re.sub(r'\s', '-', fname)
            fname = re.sub(r'[_^()\][{}]', '_', fname)
            fname = re.sub(r'[$"]', '', fname)
    path = f'out/{fname}.{ext}'
    print(path)
    plt.savefig(path, dpi=dpi)
    if NOSHOW:
        plt.close()
    else:
        plt.show()


def alpha_cmap(cmap=cm.Greys):
    my_cmap = cmap(np.arange(cmap.N))
    for i, α in enumerate(np.linspace(0, 1, cmap.N)):
        my_cmap[i, -1] = α
    return ListedColormap(my_cmap)


def rgb_cmap(r, g, b):
    return ListedColormap([[r, g, b, 1]]*32)
