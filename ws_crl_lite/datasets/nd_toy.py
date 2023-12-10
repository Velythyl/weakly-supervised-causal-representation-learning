import functools

import torch
from torch.distributions import Normal

from ws_crl.encoder import FlowEncoder
from ws_crl_lite.datasets.dataset import WSCRLDataset
from ws_crl_lite.datasets.intervset import IntervSet, IntervTable

class ToyNDDataset(WSCRLDataset):
    def __init__(self, num_samples, G, links, unlinks, intervset):
        adj_mat = nx.adjacency_matrix(G)
        adj_mat = adj_mat.toarray()
        self.adj_mat = torch.from_numpy(adj_mat)
        self.G = G

        self.links = links
        self.unlinks = unlinks
        self.intervset = intervset

        nodes = list(G.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes)}
        index_to_node = {index: node for index, node in enumerate(nodes)}
        self.node_to_index = node_to_index
        self.index_to_node = index_to_node

        self.num_nodes = adj_mat.shape[0]
        self.markov = intervset.markov

        self.execution_order = [node_to_index[n] for n in nx.topological_sort(G)]

        ALL_INTERVENTIONS = [self.intervset.init(num_samples)] # 1 is batch_size (here we are generating point-by-point, so it's 1
        for m in range(self.markov-1):
            ALL_INTERVENTIONS.append(
                self.intervset.pick(ALL_INTERVENTIONS[-1])
            )
        self.intervention_ids = torch.stack([i.self for i in ALL_INTERVENTIONS]).T

        # this messed up looking 3-for-loop piece of shit just builds one-hots
        interventions = []
        for s in range(num_samples):
            ret = []
            for m in range(self.markov):
                vec = torch.zeros(self.num_nodes).int()
                set_of_intervened_nodes = self.intervset.id2interv(self.intervention_ids[s,m])
                for n in set_of_intervened_nodes:
                    vec[n] = 1
                ret.append(vec)
            interventions.append(torch.stack(ret))
        interventions = torch.stack(interventions)
        self.interventions = interventions

        self.flow_encoder = FlowEncoder(input_features=self.num_nodes, output_features=self.num_nodes, transform_blocks=5)

        super().__init__(num_samples)


    def children(self, id):
        row = self.adj_mat[id].squeeze()
        return row.nonzero()

    def parents(self, id):
        col = self.adj_mat[:,id].squeeze()
        return col.nonzero()

    def resolve_interv(self, interv):
        # interv is set of ints (node ids) {i,j,k...}
        interv_names = {self.index_to_node[i] for i in interv}
        return interv_names

    @functools.lru_cache
    def unreachables(self, sources):
        unreachables = set(range(self.num_nodes))  # assume all unreachable
        for node in sources:
            reachable_from_node = set([self.node_to_index[n] for n in nx.descendants(self.G, self.index_to_node[node])])
            unreachables -= reachable_from_node
            unreachables -= set([node])
        return unreachables

    def gen_one(self, m_interv_sets):
        def sample_node(node_id, parent_data):
            node_name = self.index_to_node[node_id]
            link = self.links[node_name]
            ret = link(parent_data)
            return ret

        def sample_node_interv(node_id):
            return self.unlinks[self.index_to_node[node_id]]()

        vec = torch.zeros(self.num_nodes)
        for node in self.execution_order:
            parents = self.parents(node)
            if len(parents) >= 0:
                parent_data = vec[parents]
            else:
                parent_data = None
            vec[node] = sample_node(node, parent_data)

        data = [vec]

        for m in range(1,self.markov+1,1):
            data += [torch.clone(data[m-1])]

            set_of_intervened_nodes = m_interv_sets[m-1]
            if torch.sum(set_of_intervened_nodes) == 0:
                continue    # no intervention

            set_of_intervened_nodes = tuple(set_of_intervened_nodes.nonzero().unique().cpu().numpy())

            for node in self.execution_order:
                if node in self.unreachables(set_of_intervened_nodes):
                    continue

                if node in set_of_intervened_nodes:
                    val = sample_node_interv(node)
                else:
                    val = sample_node(node, data[-1][self.parents(node)])

                data[m][node] = val

        return data

    def generate(self):
        latents = []
        observations = []
        for i in range(self.num_samples):
            lat = self.gen_one(self.interventions[i])
            lat = torch.stack(lat)
            observations.append(self.flow_encoder(lat)[0])
            latents.append(lat)

        latents = torch.stack(latents)
        observations = torch.stack(observations) #self.flow_encoder(latents) # todo

        return latents, observations, self.interventions, self.intervention_ids


if __name__ == "__main__":
    import networkx as nx


    G = nx.DiGraph()

    # Add edges to the graph based on the structure you provided
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    G.add_edges_from(edges)


    adj_mat = nx.adjacency_matrix(G)
    # Convert the adjacency matrix to a NumPy array (if needed)
    adj_mat = adj_mat.toarray()

    links = {
        'A': lambda parents: Normal(0.0, 1.0).sample(),
        'B': lambda parents: Normal(0.3 * parents[0] ** 2 - 0.6 * parents[0], 0.8 ** 2).sample(),
        'C': lambda parents: Normal(0.2 * parents[0] ** 2 + -0.8 * parents[1], 1.0).sample()
    }
    # TODO define standard non-linked dists
    unlinks = {
        'A': lambda : links['A'](None),
        'B': lambda : Normal(0.4, 1.0).sample(),
        'C': lambda : Normal(-0.3, 1.0).sample()
    }

    x = IntervSet(adj_mat, 2)
    import numpy as np

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

    switch_case = IntervTable(dict_of_tables, dict_of_alphas)
        #0: Uniform(no_replace=True),
        #2: Table(dict_of_tables, dict_of_alphas)


    x.set_tables(switch_case)

    dataset = ToyNDDataset(100, G, links, unlinks, intervset=x)


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
    #do_plot(dataset.latents, dataset.intervention_ids.squeeze(), "black")
    #do_plot(dataset.observations, dataset.intervention_ids)
    exit()
