# Visualization Tool

Make sure you have the additional package requirements:

```bash
pip install fire dash plotly
```

Generate the data using `nd_toy.py` (or used the cached one in `<base repo dir>/data`:

```bash
python <base repo dir>/ws_crl_lite/nd_toy.py --data_file="./nd_toy_dataset.pt" --graph_file="./nd_toy_dataset_graph.pkl
```

Then, run the app with the following command (point to the created dataset file):

```bash
python viz_3d.py --data_file="../data/nd_toy_dataset.pt" --graph_file="../data/nd_toy_dataset_graph.pkl" --port=8899 --loglevel=DEBUG
```

Args are as follows:

```
Args:
    data_file (str): Path to pt file saved by nd_toy.py.
    graph_file (str, optional): Not implemented yet. Defaults to None.
    host (str, optional): Host url for the app; should leave it as default.
    port (str, optional): What port to serve the app on.
    loglevel (str, optional): One of DEBUG, INFO, WARN, ERROR. Defaults to "INFO".
```

Make sure it's a 3d dataset! This app won't work otherwise.
If you're on the cluster, you'll need to port forward. From your local computer,
run something like:

```bash
ssh -t -t mila -L 8899:localhost:8899 ssh <mila username>@cn-g003 -L 8899:localhost:8899
```

where you might replace `cn-g003` with the appropriate worker node you are running 
on, and `8899` with the port you've specified previously. You can then view the
app in your local browser at http://localhost:8899/

Note the following legend:

- red: z1
- green: z2
- blue: z3
- red line: z1 -> z2
- blue line: z2 -> z3

