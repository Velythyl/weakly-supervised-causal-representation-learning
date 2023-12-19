import dataclasses
import numpy as np
import fire
import pickle

import networkx as nx
import torch
from networkx import NetworkXNoCycle
from torch.distributions import Normal
from torch.utils.data import Dataset

import matplotlib
from typing import Type
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from ws_crl_lite.datasets.generate_for_graph import generate, node_to_index, roots
from ws_crl_lite.datasets.intervset import IntervSet, IntervTable
from ws_crl_minimal.encoder import FlowEncoder


class NONATOMIC_MARKOV2:
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        # FIRST, CREATE A GRAPH
        self.G = nx.DiGraph()

        # Add edges to the graph
        self.edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
        self.G.add_edges_from(self.edges)

        self.x = IntervSet(self.G, 2)

        # GIVEN THE PRINTED STATEMENT ABOVE, YOU CAN DEFINE YOUR TABLES.
        # (it's also easy to automate this using a forloop on the markov length)
        self.dict_of_tables = {
            0: np.ones(self.x.num_interv_ids),
            1: np.random.uniform(0, 10, size=(self.x.num_interv_ids, self.x.num_interv_ids)),
            2: np.random.uniform(0, 10, size=(self.x.num_interv_ids, self.x.num_interv_ids))
        }
        self.alpha_vec = np.random.uniform(0.1,1, size=(3,))
        # fiself.xme
        # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
        self.switch_case = IntervTable(self.dict_of_tables, self.alpha_vec)

        self.x.set_tables(self.switch_case)
        self.x.kill(intervs_of_size=2, intervs_in_set={1:[3], 2: [1]})

        # DEFINE THE RELATIONSHIP OF EACH NODE TO ITS PARENT
        # (to automate this, just an affine transform given the parents)
        self.links = {
            'A': lambda parents: Normal(0.0, 1.0).sample(),
            'B': lambda parents: Normal(0.3 * parents[0] ** 2 - 0.6 * parents[0], 0.16).sample(),
            'C': lambda parents: Normal(0.2 * parents[0] ** 2 + -0.8 * parents[1], 1.0).sample()
        }

        # DEFINE HOW THE NODES BEHAVE WHEN THEY GET INTERVENED ON
        # (to automate this, just sample from a normal or something of the sort)
        self.unlinks = {
            'A': lambda: self.links['A'](None),
            'B': lambda: Normal(0.4, 1.0).sample(),
            'C': lambda: Normal(-0.3, 1.0).sample()
        }
    
    def dataset_kwargs(self):
        return {
            "timesteps": 2,
            "G": self.G,
            "links": self.links,
            "unlinks": self.unlinks,
            "intervset": self.x
        }

class ATOMIC_MARKOV1:
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        # FIRST, CREATE A GRAPH
        self.G = nx.DiGraph()

        # Add edges to the graph
        self.edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
        self.G.add_edges_from(self.edges)

        self.x = IntervSet(self.G, markov=1)  #, set_of_all_intervs=[(), (0,), (1,), (2,)])

        # GIVEN THE PRINTED STATEMENT ABOVE, YOU CAN DEFINE YOUR TABLES.
        # (it's also easy to automate this using a forloop on the markov length)
        # self.dict_of_tables = {
        #     0: np.ones(self.x.num_interv_ids),
        #     1: np.random.uniform(0, 10, size=(self.x.num_interv_ids, self.x.num_interv_ids)),
        #     2: np.random.uniform(0, 10, size=(self.x.num_interv_ids, self.x.num_interv_ids))
        # }

        self.dict_of_tables = {
            0: np.ones(self.x.num_interv_ids), 
        }

        self.alpha_vec = np.random.uniform(0.1,1, size=(2,))

        # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
        self.switch_case = IntervTable(self.dict_of_tables, self.alpha_vec)

        self.x.set_tables(self.switch_case)
        self.x.kill(intervs_of_size=2)  #, intervs_in_set={1:[3], 2: [1]})
        self.x.kill(intervs_of_size=3)  #, intervs_in_set={1:[3], 2: [1]})

        # DEFINE THE RELATIONSHIP OF EACH NODE TO ITS PARENT
        # (to automate this, just an affine transform given the parents)
        self.links = {
            'A': lambda parents: Normal(0.0, 1.0).sample(),
            'B': lambda parents: Normal(0.3 * parents[0] ** 2 - 0.6 * parents[0], 0.16).sample(),
            'C': lambda parents: Normal(0.2 * parents[0] ** 2 + -0.8 * parents[1], 1.0).sample()
        }

        # DEFINE HOW THE NODES BEHAVE WHEN THEY GET INTERVENED ON
        # (to automate this, just sample from a normal or something of the sort)
        self.unlinks = {
            'A': lambda: self.links['A'](None),
            'B': lambda: Normal(0.4, 1.0).sample(),
            'C': lambda: Normal(-0.3, 1.0).sample()
        }
    
    def dataset_kwargs(self):
        return {
            "timesteps": 1,
            "G": self.G,
            "links": self.links,
            "unlinks": self.unlinks,
            "intervset": self.x
        }

GRAPH_DEFS = {
    "nonatomic_markov2": NONATOMIC_MARKOV2,
    "atomic_markov1": ATOMIC_MARKOV1,
}


def maybe_detach(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach()
    return arr


@dataclasses.dataclass
class WSCRLData:
    latents: torch.Tensor
    observations: torch.Tensor


class WSCRLDataset(Dataset):
    def __init__(self, num_samples, timesteps, G, links, unlinks, intervset, timestep_carryover=True):
        self.num_samples = num_samples
        self.intervset = intervset
        self.G = G

        self.latents, self.observations, self.interventions, self.intervention_ids = generate(
            num_samples, 
            timesteps, 
            G,
            links, 
            unlinks, 
            intervset,
            timestep_carryover=timestep_carryover
        )

        self.latents = maybe_detach(self.latents)
        self.observations = maybe_detach(self.observations)
        self.interventions = maybe_detach(self.interventions)
        self.intervention_ids = maybe_detach(self.intervention_ids)

    @property
    def markov(self):
        return self.intervset.markov

    @property
    def num_interv_types(self):
        # fixme big assumption: all interventions were sampled during generation (this might not be true)
        return self.intervention_ids.unique().shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.observations[idx], self.latents[idx], self.intervention_ids[idx], self.interventions[idx]

    def __eq__(self, other):
        # Check for equality of probability tables
        for (k1, v1), (k2, v2) in zip(
            self.intervset.probability_tables.dict_of_tables.items(), 
            other.intervset.probability_tables.dict_of_tables.items()
        ):
            if not (v1 == v2).all():
                return False
        # Check equality of alpha vec
        if (self.intervset.probability_tables.alpha_vec != other.intervset.probability_tables.alpha_vec).all():
            return False
        # Check equality of adj matrix
        if (self.intervset.adj_mat != other.intervset.adj_mat).all():
            return False
        return True

class AutomaticDataset(WSCRLDataset):
    def __init__(self, num_samples, timesteps, markov, G, timestep_carryover, seed: int = None):
        try:
            nx.find_cycle(G)
            raise AssertionError("Graph can't have cycles")
        except NetworkXNoCycle:
            pass
    
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        n2i = node_to_index(G)
        starts = roots(G)
        descendents = set(list(G.nodes)) - starts
        assert len(starts) != 0

        links = {k: lambda parents: Normal(n2i[k], 1.0).sample() for k in starts}
        unlinks = {k: lambda: v(None) for k, v in links.items()}

        for node in descendents:
            # find number of parents
            n_parents = len(list(G.predecessors(node)))

            def make_link():
                flow_encoder = FlowEncoder(
                    input_features=n_parents,
                    output_features=n_parents,
                    transform_blocks=2  # todo maybe add blocks
                )

                def descendant_link(parents):
                    flow = flow_encoder(parents[None])[0]

                    if n2i[node] % 2 == 0:
                        flow = flow.mean()
                    else:
                        flow = flow.sum()

                    return Normal(0.0, 1.0).sample() + flow

                return descendant_link

            links[node] = make_link()
            unlinks[node] = lambda: Normal(0.1 * n2i[node], 1.0).sample()

        intervset = IntervSet(G, markov)

        def random_uniform(is_vec):
            return np.random.uniform(0, 10, size=(intervset.num_interv_ids,) if is_vec else (
            intervset.num_interv_ids, intervset.num_interv_ids))

        dict_of_tables = {i: random_uniform(False) for i in range(markov + 1)}
        dict_of_tables[0] = random_uniform(True)

        alpha_vec = np.random.uniform(0.1, 1, size=(markov+1))

        # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
        switch_case = IntervTable(dict_of_tables, alpha_vec)
        intervset.set_tables(switch_case)

        super().__init__(num_samples, timesteps, G, links, unlinks, intervset, timestep_carryover)


def n_node_dataset(
    num_datasets, 
    num_nodes_OR_generator, 
    num_samples, 
    timesteps, 
    markov, 
    seed: int = 42
):
    torch.manual_seed(seed)

    if isinstance(num_nodes_OR_generator, int):
        if num_nodes_OR_generator == 2:
            def gen_graph(_):
                # Fixed 2 node graph
                G = nx.DiGraph()
                edges = [('A', 'B')]
                G.add_edges_from(edges)
                return G

        def has_cycle(g):
            try:
                nx.find_cycle(g)
                return True
            except:
                return False

        def gen_graph(i):
            def gen():
                return nx.fast_gnp_random_graph(num_nodes_OR_generator, 0.7, directed=True)

            g = gen()
            while has_cycle(g):
                g = gen()

            return g

        generator = gen_graph
    else:
        generator = num_nodes_OR_generator

    ret = []
    while len(ret) != num_datasets:
        graph = generator(len(ret))
        ret += [AutomaticDataset(num_samples, timesteps, markov, graph, timestep_carryover=False)]
    return ret


def build_train_test_n_node_dataset(
    n_train, 
    n_test,
    num_nodes_OR_generator, 
    timesteps, 
    markov, 
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(num_nodes_OR_generator, int):
        def has_cycle(g):
            try:
                nx.find_cycle(g)
                return True
            except:
                return False

        def gen_graph(i):
            def gen():
                return nx.fast_gnp_random_graph(num_nodes_OR_generator, 0.7, directed=True)

            g = gen()
            while has_cycle(g):
                g = gen()

            return g

        generator = gen_graph
    else:
        generator = num_nodes_OR_generator

    graph = generator(0)  # ??
    train = AutomaticDataset(n_train, timesteps, markov, graph, timestep_carryover=False, seed=seed)
    test = AutomaticDataset(n_test, timesteps, markov, graph, timestep_carryover=False, seed=seed)
    if train != test:
        raise RuntimeError("train and test datasets have different prob tables or adj matricies")
    return train, test


def build_manual_datasets(n_samples, graph_def: str = "nonatomic_markov2", seed: int = 42):
    """Builds graph using manual graph defs
    If n_samples is a list, make a dataset for every one of the specified lengths
    """

    if graph_def not in GRAPH_DEFS:
        raise ValueError(f"graph_df is {graph_def}; must be one of {list(GRAPH_DEFS.keys())}")

    if seed is not None:
        torch.manual_seed(seed=seed)
    graph_def = GRAPH_DEFS[graph_def](seed)

    if not isinstance(n_samples, list):
        n_samples = [n_samples]
    datasets = [WSCRLDataset(n, **graph_def.dataset_kwargs()) for n in n_samples]
    return datasets


def jank_main(
    data_file: str = "nd_toy_dataset.pt", 
    graph_file: str = "nd_toy_dataset_graph.pkl",
    n_samples: int = 10000,   
    auto: bool = True,
):
    # FIRST, CREATE A GRAPH
    G = nx.DiGraph()
    # Add edges to the graph
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    G.add_edges_from(edges)

    if auto:
        dataset = n_node_dataset(4, 3, num_samples=n_samples, timesteps=2, markov=2)[0]
    else:
        raise NotImplementedError
        # dataset = build(G, n_samples)

    def plot_3d(data):
        interventions = dataset.intervention_ids
        min_interv = interventions[interventions != 0].min()
        max_interv = interventions[interventions != 0].max()

        cmap = cm.viridis
        norm = Normalize(vmin=min_interv, vmax=max_interv)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        color = {
            0: "red",
            1: "green",
            2: "blue",
        }
        for i in range(data.shape[1]):
            ar = data[:, i]
            ax.scatter(ar[:, 0], ar[:, 1], ar[:, 2], color=color[i])

        def plot_many_arrows(pairs, color):
            # pairs is of shape [n arrows, 2, (x,y)]

            base_x = pairs[:, 0, 0]
            base_y = pairs[:, 0, 1]
            base_z = pairs[:, 0, 2]

            end_x = pairs[:, 1, 0]
            end_y = pairs[:, 1, 1]
            end_z = pairs[:, 1, 2]
            dx = end_x - base_x
            dy = end_y - base_y
            dz = end_z - base_z

            for i in range(base_x.shape[0]):
                ax.plot([base_x[i], end_x[i]], [base_y[i], end_y[i]], [base_z[i], end_z[i]], color=color)

        def plot_intervs(data, interventions):
            NUM_INTERVS_OF_EACH_TYPE_TO_PLOT = 2

            for i in interventions.unique():
                if i == 0:
                    continue
                # if i not in list(range(dataset.num_nodes+1)): # skips intervs on more than one node
                #    continue

                # opt to select the first elements. doesn't change anything anyway.
                selected_intervs = torch.argsort(interventions == i, descending=True)
                selected_intervs = selected_intervs[:NUM_INTERVS_OF_EACH_TYPE_TO_PLOT].squeeze()
                assert (interventions[selected_intervs] == i).all()

                sel_latents = data[selected_intervs]
                plot_many_arrows(sel_latents, color=cmap(norm(i)))

        plot_intervs(data[:, :2, :], dataset.intervention_ids[:, 0].squeeze())
        if dataset.markov == 2:
            plot_intervs(data[:, 1:, :], dataset.intervention_ids[:, 1].squeeze())
        plt.show()
    
    data = dataset[:]
    intervention_labels = np.packbits(data[3], bitorder='big', axis=2) >> (8 - data[3].shape[2])
    torch.save((*data[:2], data[3], intervention_labels.squeeze()), data_file)
    with open(graph_file, "wb") as f:
        pickle.dump(dataset.G, f)

    plot_3d(dataset.latents)
    plot_3d(dataset.observations)
    # do_plot(dataset.latents, dataset.intervention_ids.squeeze(), "black")
    # do_plot(dataset.observations, dataset.intervention_ids)
    exit()


if __name__ == "__main__":
    fire.Fire(jank_main)
