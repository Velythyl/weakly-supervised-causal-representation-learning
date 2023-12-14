# PGM Group Project Extension

### Changelog

| Date | User | Changes |
|------|------|---------|
| 22.12.13 | David | added the `visualizations` dir for interactive viz of data gen by `ws_crl_lite/dataset/nd_toy.py` |
| 22.12.08 | David | taking the original code and scrubbing everything not related to the ILCM experiments |


## Installation

**Pre-installation if you're a loser**

Basically follow instructions on the [Mila docs](https://docs.mila.quebec/Userguide.html#Mamba) (search for Mamba).
Copy pasted here because I'm nice.


When installing new packages with conda install, conda uses a built-in dependency solver for solving the dependency graph of all packages (and their versions) requested such that package dependency conflicts are avoided.

In some cases, especially when there are many packages already installed in a conda environment, conda’s built-in dependency solver can struggle to solve the dependency graph, taking several to tens of minutes, and sometimes never solving. In these cases, it is recommended to try libmamba.

To install and set the libmamba solver, run the following commands:

```bash
# Install miniconda
# (you can not use the preinstalled anaconda/miniconda as installing libmamba
#  requires ownership over the anaconda/miniconda install directory)
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
bash Miniconda3-py310_22.11.1-1-Linux-x86_64.sh

# Install libmamba
conda install -n base conda-libmamba-solver
```

By default, conda uses the built-in solver when installing packages, even after installing other solvers. To try `libmamba` once, add `--solver=libmamba` in your `conda install` command. For example:

```bash
conda install tensorflow --solver=libmamba
```

You can set libmamba as the default solver by adding solver: libmamba to your .condarc configuration file located under your `$HOME` directory. You can create it if it doesn’t exist. You can also run:

```bash
conda config --set solver libmamba
```


**Environment setup**

1. I used conda, glhf

```bash
conda env create --file env.yaml
```

2. Activate it

```bash
conda activate weakly-supervised-causal
```

3. Test it with a base command

```bash
experiments/scaling.py general.exp_name=dummy general.base_dir="./results_test/"  general.seed=42 data.dim_z=3 data.nature.seed=42 training=scaling_fast
```


# Minimal Working WS CRL 

All scrubbed code is in `*_minimal/` folders.
More info TBD.

### Tracking

To start the mlflow server, run

```bash
mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db
```

wherever the `mlflow.db` is stored (e.g. in the `run_*/` folders).
You'll have to port forward on your personal computer, e.g. something like:

```
ssh -t -t mila -L 8080:localhost:8080 ssh <USER>@cn-g007 -L 8080:localhost:8080
```

and then open on your home machine: http://localhost:8080.

&nbsp;
&nbsp;
&nbsp;
&nbsp;

---

&nbsp;
&nbsp;
&nbsp;
&nbsp;

# ORIGINAL README: Weakly supervised causal representation learning

This repository contains the code for the paper [**Weakly supervised causal representation learning**](https://arxiv.org/abs/2203.16437) by Johann Brehmer, Pim de Haan, Phillip Lippe, and Taco Cohen, published at NeurIPS 2022.

## Abstract

Learning high-level causal representations together with a causal model from unstructured low-level data such as pixels is impossible from observational data alone. We prove under mild assumptions that this representation is however identifiable in a weakly supervised setting. This involves a dataset with paired samples before and after random, unknown interventions, but no further labels. We then introduce implicit latent causal models, variational autoencoders that represent causal variables and causal structure without having to optimize an explicit discrete graph structure. On simple image data, including a novel dataset of simulated robotic manipulation, we demonstrate that such models can reliably identify the causal structure and disentangle causal variables.

## Using the repository

### Repository structure

```
weakly-supervised-causal-representation-learning
│   .gitignore
│   Dockerfile: specification of a Docker image that provides all dependencies
│   LICENSE
│   README.md: this README file
│   setup.py
│
└───config: configuration YAML files for the experiments we ran
|
└───experiments: experiment scripts
│   |   causalcircuit.py: experiment script for the CausalCircuit dataset after its dimensionality has been reduced
│   |   causalcircuit_reduce_dim.py: experiment script to run dimensionality reduction on the CausalCircuit dataset
|   |   experiment_utils.py: helper functions for experiment scripts
│   |   scaling.py: experiment script for the graph-scaling toy experiments
|
└───ws_crl: implementation of models and utilities for this project
    │   distributions.py: probability distributions
    │   gumbel.py: differentiable parameterizations of discrete distributions
    │   metrics.py: evaluation metrics
    │   nets.py: neural network layers
    │   plotting.py: helper functions for plotting
    │   splines.py: spline-based transformations
    │   training.py: training objectives and metrics
    │   transforms.py: invertible transformations for use in flows
    │   utils.py: utility functions
    |
    └───causal: components that describe causal structure between causal variables
    |   | graph.py: graph parameterizations
    |   | implicit_scm.py: implicit parameterization of causal structure through solution functions
    |   | interventions.py: distributions over intervention targets
    |   | scm.py: explicit parameterization of causal structure through causal mechanisms
    |
    └───encoder: encoder and decoder
    |   | base.py: base encoder / decoder classes
    |   | flow.py: invertible transformations for flow models (used in ground-truth SCMs)
    |   | image_vae.py: VAEs for image data
    |   | vae.py: VAEs for scalar data
    |
    └───lcm: LCMs (which combine an SCM, an encoder, and a decoder)
    |   | base.py: base LCM class
    |   | elcm.py: explicit LCMs, VAEs with causal variables as latents and an SCM in the latent space
    |   | flow_lcm.py: LCM using a flow-style invertible transformation from causal variables to data space
    |   | ilcm.py: implicit LCMs, VAEs with noise variables as latents and an implicit representation of causal structure
    |
    └───posthoc_graph_learning: graph inference from ILCMs
        | enco.py: ENCO-based causal discovery
        | implicit.py: heuristic causal discovery
```

### Getting started

First, let's define paths for the code, the experiments, and the dataset. Adapt as needed.
```
CRLREPO=$(realpath .)/wscrl/repo
CRLDATA=$(realpath .)/wscrl/datasets
mkdir --parents $CRLDATA
```

Clone the repository:
```
git clone https://github.com/Velythyl/weakly-supervised-causal-representation-learning $CRLREPO
cd $CRLREPO
```

Build the provided Docker image and run the container in attached mode, while mounting the data folder:
```
docker build --tag wscrl:latest .
docker run -v "$CRLDATA:/workspace/data" --gpus all --name wscrl -it --rm wscrl:latest
```

Now you can run experiments, which we will discuss in the next section in more detail. To quickly verify that the code is running, use this dummy experiment:
```
experiments/scaling.py general.exp_name=dummy general.base_dir=results_whatever general.seed=42 data.dim_z=3 data.nature.seed=42 training=scaling_fast
```
You should be seeing log output from the training, including a progress bar.

After a few minutes, training should finish; you should find the final message `Anders nog iets?` in the output. Directly above it, you should see a statement like `Final evaluation: causal disentanglement = 0.99`, which tells you that the ILCM disentangled the causal variables successfully. Note that the code is [not fully deterministic](https://pytorch.org/docs/stable/notes/randomness.html), so the result may vary slightly run by run.

As a final check, you can inspect the test metrics:
```
cat /workspace/exp/dummy/metrics/test_metrics.csv
```

### Running toy experiments

This repository contains code to run experiments on two datasets: the *scaling* dataset (Section 5.4 in the paper) and the *CausalCircuit* dataset (Section 5.3).

The scaling dataset consists of `<DATA-DIM>`-dimensional toy data with random graphs and is generated on the fly. Running experiments should be straightforward:

```
experiments/scaling.py general.exp_name=<EXP_NAME> general.base_dir=<PATH/TO/ROOT_DIRECTORY> general.seed=<MODEL_SEED> data.dim_z=<DATA_DIM> data.nature.seed=<DATA_SEED>
```
where `<MODEL_SEED>` and `<DATA_SEED>` determine the initialization of the model and the dataset, respectively, and `<DATA_DIM>` is the desired dimensionality of the dataset (which in this dataset is equal to the number of causal variables). For instance:
```
experiments/scaling.py general.exp_name=scaling_ilcm_4_vars general.base_dir=/workspace/exp general.seed=4901 data.dim_z=4 data.nature.seed=302
```

By default, these scripts run experiments with our ILCM method. You can switch to baselines by passing `--config-name=scaling_betavae` or `--config-name=scaling_dvae`. (These options are shorthand for switching the model and training configs to the correct settings.) For example:
```
experiments/scaling.py --config-name=scaling_betavae general.exp_name=scaling_betavae_4_vars general.base_dir=/workspace/exp general.seed=4901 data.dim_z=4 data.nature.seed=302
experiments/scaling.py --config-name=scaling_dvae general.exp_name=scaling_dvae_4_vars general.base_dir=/workspace/exp general.seed=4901 data.dim_z=4 data.nature.seed=302
```

Experiments log their results to an MLflow SQLite database specified in the `general.mlflow.db` config entry, as well as to the output folder in `<EXPERIMENT_FOLDER>`, which is specified in the config entry `general.exp_dir`. In particular, you can find evaluation results in `<EXPERIMENT_FOLDER>/metrics/test_metrics.csv`.

To reproduce the results in figure 8 of the paper, you will need to run each of these three models for each value of `<DATA_DIM>` with nine different configurations: three different values for `<MODEL_SEED>` times three different values for `<DATA_SEED>`. We tested the three concrete examples above and found the following disentanglement scores:
``` 
   Method | disentanglement score
----------+-----------------------
     ILCM |                  0.99
     dVAE |                  0.34
 beta-VAE |                  0.13
```
Note that the code is [not fully deterministic](https://pytorch.org/docs/stable/notes/randomness.html), you may thus find deviations (though after avereging over multiple seeds these should be negligible).


### Running CausalCircuit experiments

The CausalCircuit dataset shows a robot arm interacting with a causally connected circuit of buttons and lights. The data  consists of 512x512 images.

Working with the CausalCircuit dataset consists of three steps: downloading the dataset, training a dimensionality reduction model, and then training causal models and baselines on dimensionality-reduced data.

#### 1. Downloading the dataset

First, download the dataset from [developer.qualcomm.com/software/ai-datasets/causalcircuit](https://developer.qualcomm.com/software/ai-datasets/causalcircuit). Store the dataset under `$CRLDATA/causalcircuit` (or adapt the paths below).


#### 2. Training a dimensionality reduction model

Run
```
experiments/causalcircuit_reduce_dim.py general.exp_name=<DIM_RED_EXP_NAME> general.base_dir=<PATH/TO/ROOT_DIRECTORY> general.seed=<SEED> data.data_dir=<DATA_DIR>
```

For instance:
```
experiments/causalcircuit_reduce_dim.py general.exp_name=causalcircuit_reduce_dim general.base_dir=/workspace/exp general.seed=4601 data.data_dir=/workspace/data/causalcircuit
```

If you encounter GPU out of memory issues, reduce the default batch size of 32, by adding e.g. `training.batchsize=8`. In this case, we recommend to also lower the learning rate (the increased stochasticity from smaller batchsize can otherwise lead to issues), for instance by providing the argument `training.lr_schedule.initial=3e-5`. Instead of specifying these settings individually, you can also use the`training=causalcircuit_reduce_dim_fast`, which uses a smaller batchsize, lower learning rate, and shorter training, for quicker experimentation (though the quality of the learned representations may be different from the ones we used in the paper). 

#### 3. Training causal models and baselines on dimensionality-reduced data


Run
```
experiments/causalcircuit.py general.exp_name=<EXP_NAME> general.base_dir=<PATH/TO/ROOT_DIRECTORY> data.data_dir=<PATH/TO/DIM_RED_MODEL>/data  general.seed=<SEED>
```

where `<PATH/TO/DIM_RED_MODEL>` is the path to the directory in which the dimensionality-reduction experiment is stored &#40;by default `<PATH/TO/ROOT_DIRECTORY>/experiments/<DIM_RED_EXP_NAME>`&#41;. For instance:

```
experiments/causalcircuit.py general.exp_name=causalcircuit_ilcm general.base_dir=/workspace/exp data.data_dir=/workspace/exp/causalcircuit_reduce_dim/data  general.seed=4701
```

By default, these scripts run experiments with our ILCM method. You can switch to baselines by passing `--config-name=causalcircuit_betavae` or `--config-name=causalcircuit_dvae`.


## Citation

If you find our code useful, please cite:

```
@inproceedings{brehmer2022weakly,
  title = {Weakly supervised causal representation learning},
  author = {Brehmer, Johann and De Haan, Pim and Lippe, Phillip and Cohen, Taco},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2022},
  volume = {35},
  eprint = {2203.16437},
  url = {https://arxiv.org/abs/2203.16437},
}
```
