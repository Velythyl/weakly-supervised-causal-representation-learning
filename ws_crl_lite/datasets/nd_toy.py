import functools

import torch
from torch.distributions import Normal
from tqdm import tqdm

from ws_crl.encoder import FlowEncoder
from ws_crl_lite.datasets.dataset import WSCRLDataset
from ws_crl_lite.datasets.intervset import IntervSet, IntervTable

def list_difference(list1, list2):
    list2 = set(list2)
    return [item for item in list1 if item not in list2]

@functools.lru_cache
def node_to_index(G):
    nodes = list(G.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    return node_to_index

@functools.lru_cache
def index_to_node(G):
    nodes = list(G.nodes())
    index_to_node = {index: node for index, node in enumerate(nodes)}
    return index_to_node

@functools.lru_cache
def reachables(G, source_ids):
    source_names = map(lambda i: index_to_node(G)[i], source_ids)

    reachables = set()
    for node in source_names:
        reachable_from_node = set(nx.descendants(G, node))

        reachables.update(reachable_from_node)
        reachables.add(node)

    return list(map(lambda i: node_to_index(G)[i], reachables))

@functools.lru_cache
def parents_dict(G):
    n2i = node_to_index(G)

    ret = {}
    for n, i in n2i.items():
        ret[i] = torch.tensor([n2i[parent] for parent in G.predecessors(n)]).int()
    return ret

@functools.lru_cache
def execution_order(G):
    return [node_to_index(G)[n] for n in nx.topological_sort(G)]

@functools.lru_cache
def intervened_execution_order(G, intervened_nodes):
    ret = []
    for i in execution_order(G):
        if i in reachables(G, intervened_nodes):
            ret += [i]
    return ret

def num_nodes(G):
    return len(G.nodes())


def generate_one(interventions, G, links, unlinks):
    def sample_node(node_id, parent_data):
        link = links[node_id]
        ret = link(parent_data)
        return ret

    def sample_node_interv(node_id):
        return unlinks[node_id]()

    vec = torch.zeros(num_nodes(G))
    for node in execution_order(G):
        parents = parents_dict(G)[node]
        if len(parents) >= 0:
            parent_data = vec[parents]
        else:
            parent_data = None
        vec[node] = sample_node(node, parent_data)

    data = [vec]

    for m in range(1, len(interventions)+1, 1):
        data += [torch.clone(data[m-1])]

        set_of_intervened_nodes = interventions[m-1]
        if len(set_of_intervened_nodes) == 0:
            continue

        #ordered_affected_nodes = list_difference(self.execution_order, self.unreachables(set_of_intervened_nodes))
        for node in intervened_execution_order(G, set_of_intervened_nodes):
            if node in set_of_intervened_nodes:
                val = sample_node_interv(node)
            else:
                val = sample_node(node, data[-1][parents_dict(G)[node]])

            data[m][node] = val
    return data

def generate(num_samples, timesteps, G, links, unlinks, intervset):
    links = {node_to_index(G)[k]: v for k, v in links.items()}
    unlinks = {node_to_index(G)[k]: v for k, v in unlinks.items()}
    num_nodes = len(G.nodes())

    flow_encoder = FlowEncoder(
        input_features=num_nodes,
        output_features=num_nodes,
        transform_blocks=5
    )
    
    
    ALL_INTERVENTIONS = [intervset.init(num_samples)]  # 1 is batch_size (here we are generating point-by-point, so it's 1
    for m in range(timesteps):
        ALL_INTERVENTIONS += [intervset.pick(ALL_INTERVENTIONS[-1])]
    intervention_ids = torch.stack([i.self for i in ALL_INTERVENTIONS]).T
    interventions = intervset.n_m_onehots(intervention_ids)

    latents = []
    observations = []
    intervention_tuples = intervset.onehots_to_tuples(interventions)

    for i in tqdm(range(num_samples)):
        lat = generate_one(intervention_tuples[i], G, links, unlinks)
        lat = torch.stack(lat)
        observations.append(flow_encoder(lat)[0])
        latents.append(lat)

    latents = torch.stack(latents)
    observations = torch.stack(observations)

    return latents, observations, interventions, intervention_ids


class ToyNDDataset(WSCRLDataset):
    def __init__(self, num_samples, G, links, unlinks, intervset):
        self.G = G

        self.links = {node_to_index(G)[k]: v for k, v in links.items()}
        self.unlinks = {node_to_index(G)[k]: v for k, v in unlinks.items()}
        self.intervset = intervset

        self.num_nodes = len(G.nodes())
        self.markov = intervset.markov


        ALL_INTERVENTIONS = [self.intervset.init(num_samples)] # 1 is batch_size (here we are generating point-by-point, so it's 1
        for m in range(self.markov-1):
            ALL_INTERVENTIONS.append(
                self.intervset.pick(ALL_INTERVENTIONS[-1])
            )
        self.intervention_ids = torch.stack([i.self for i in ALL_INTERVENTIONS]).T
        self.interventions = self.intervset.n_m_onehots(self.intervention_ids)

        self.flow_encoder = FlowEncoder(
            input_features=self.num_nodes,
            output_features=self.num_nodes,
            transform_blocks=5
        )

        super().__init__(num_samples)

    def gen_one(self, m_interv_sets):
        def sample_node(node_id, parent_data):
            link = self.links[node_id]
            ret = link(parent_data)
            return ret

        def sample_node_interv(node_id):
            return self.unlinks[node_id]()

        vec = torch.zeros(self.num_nodes)
        for node in execution_order(self.G):
            parents = parents_dict(G)[node]
            if len(parents) >= 0:
                parent_data = vec[parents]
            else:
                parent_data = None
            vec[node] = sample_node(node, parent_data)

        data = [vec]

        for m in range(1,self.markov+1,1):
            data += [torch.clone(data[m-1])]

            set_of_intervened_nodes = m_interv_sets[m-1]
            if len(set_of_intervened_nodes) == 0:
                continue

            #ordered_affected_nodes = list_difference(self.execution_order, self.unreachables(set_of_intervened_nodes))
            for node in intervened_execution_order(self.G, set_of_intervened_nodes):
                if node not in reachables(self.G, set_of_intervened_nodes):
                    continue

                if node in set_of_intervened_nodes:
                    val = sample_node_interv(node)
                else:
                    val = sample_node(node, data[-1][parents_dict(G)[node]])

                data[m][node] = val

        return data

    def generate(self):
        latents = []
        observations = []

        intervention_tuples = self.intervset.onehots_to_tuples(self.interventions)

        for i in tqdm(range(self.num_samples)):
            lat = self.gen_one(intervention_tuples[i])
            lat = torch.stack(lat)
            observations.append(self.flow_encoder(lat)[0])
            latents.append(lat)

        latents = torch.stack(latents)
        observations = torch.stack(observations)

        return latents, observations, self.interventions, self.intervention_ids


if __name__ == "__main__":
    import networkx as nx

    # FIRST, CREATE A GRAPH
    G = nx.DiGraph()
    # Add edges to the graph
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    G.add_edges_from(edges)

    # COMPUTE THE SET OF INTERVENTIONS: THE INTERVSET
    x = IntervSet(G, 2)
    print(x.set_of_all_intervs)
    import numpy as np

    # GIVEN THE PRINTED STATEMENT ABOVE, YOU CAN DEFINE YOUR TABLES.
    # (it's also easy to automate this using a forloop on the markov length)
    dict_of_tables = {
        0: np.ones(x.num_interv_ids),
        1: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids)),
        2: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids))
    }
    dict_of_alphas = {
        0: [1],
        1: [1],
        2: [0.5, 1]
    }
    # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
    switch_case = IntervTable(dict_of_tables, dict_of_alphas)
    x.set_tables(switch_case)

    # DEFINE THE RELATIONSHIP OF EACH NODE TO ITS PARENT
    # (to automate this, just an affine transform given the parents)
    links = {
        'A': lambda parents: Normal(0.0, 1.0).sample(),
        'B': lambda parents: Normal(0.3 * parents[0] ** 2 - 0.6 * parents[0], 0.8 ** 2).sample(),
        'C': lambda parents: Normal(0.2 * parents[0] ** 2 + -0.8 * parents[1], 1.0).sample()
    }

    # DEFINE HOW THE NODES BEHAVE WHEN THEY GET INTERVENED ON
    # (to automate this, just sample from a normal or something of the sort)
    unlinks = {
        'A': lambda: links['A'](None),
        'B': lambda: Normal(0.4, 1.0).sample(),
        'C': lambda: Normal(-0.3, 1.0).sample()
    }


    generate(1000, 2, G, links, unlinks, intervset=x)

    # PASS THE GRAPH, THE LINKS, THE UNLINKS, AND THE INTERVSET
    from dataset2 import WSCRLDataset
    dataset = WSCRLDataset(1000, 2, G, links, unlinks, intervset=x)


    # To access a single sample
    sample = dataset[0]
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    def plot_3d(data):
        interventions = dataset.intervention_ids
        min_interv = interventions[interventions != 0].min()
        max_interv = interventions[interventions != 0].max()
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

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
            ar = data[:,i]
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
                #if i not in list(range(dataset.num_nodes+1)): # skips intervs on more than one node
                #    continue

                # opt to select the first elements. doesn't change anything anyway.
                selected_intervs = torch.argsort(interventions == i, descending=True)
                selected_intervs = selected_intervs[:NUM_INTERVS_OF_EACH_TYPE_TO_PLOT].squeeze()
                assert (interventions[selected_intervs] == i).all()

                sel_latents = data[selected_intervs]
                plot_many_arrows(sel_latents, color=cmap(norm(i)))

        plot_intervs(data[:,:2,:], dataset.intervention_ids[:,0].squeeze())
        if dataset.markov == 2:
            plot_intervs(data[:,1:,:], dataset.intervention_ids[:,1].squeeze())
        plt.show()
    plot_3d(dataset.latents)
    plot_3d(dataset.observations)
    #do_plot(dataset.latents, dataset.intervention_ids.squeeze(), "black")
    #do_plot(dataset.observations, dataset.intervention_ids)
    exit()
