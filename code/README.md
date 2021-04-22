
Generating some illustrations for the presentation.
It will produce images in the `out/` subfolder, and display them.
To disable display, prefix you commande with `BATCH=1` (see example below where it is especially important for the ensembling code that opens a lot of graphs) or hardcode it in `tools.py`.

~~~
python3 loss-landscape-2d.py
python3 dataset-1d-regression.py
python3 ensembling.py
BATCH=1 python3 ensembling.py
~~~
