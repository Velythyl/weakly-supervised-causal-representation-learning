import functools

import torch
from torch.distributions import Normal

from repo.ws_crl_lite.datasets.dataset import WSCRLDataset
from repo.ws_crl_lite.datasets.intervset import IntervSet, Uniform, Table


def has_cycle_dfs(node, adjacency_matrix, visited, parent):
    visited[node] = True

    for neighbor in range(len(adjacency_matrix)):
        if adjacency_matrix[node][neighbor]:
            if not visited[neighbor]:
                if has_cycle_dfs(neighbor, adjacency_matrix, visited, node):
                    return True
            elif parent != neighbor:
                return True

    return False

def has_cycle(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    visited = [False] * num_nodes

    for node in range(num_nodes):
        if not visited[node]:
            if has_cycle_dfs(node, adjacency_matrix, visited, -1):
                return True

    return False


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, a.T, rtol=rtol, atol=atol)

def spicy(self, node_id, prev_ids, data):
    string = self.G.get_node_attributes(self.index_to_node[node_id])

    print("good development practices <3")
    def quarantine():
        for prev in prev_ids:
            locals()[prev] = data[prev]

        retval = eval(string)
        return retval
    return quarantine()


class ToyNDDataset(WSCRLDataset):
    def __init__(self, num_samples, G, links, unlinks, intervset):
        adj_mat = nx.adjacency_matrix(G)
        # Convert the adjacency matrix to a NumPy array (if needed)
        adj_mat = adj_mat.toarray()
        self.adj_mat = torch.from_numpy(adj_mat)
        self.G = G
        self.intervset = intervset


        nodes = list(G.nodes())

        # Create a mapping between node labels and indices in the adjacency matrix
        node_to_index = {node: index for index, node in enumerate(nodes)}
        index_to_node = {index: node for index, node in enumerate(nodes)}
        self.node_to_index = node_to_index
        self.index_to_node = index_to_node

        self.links = links
        self.unlinks = unlinks

        self.num_nodes = adj_mat.shape[0]
        self.markov = intervset.markov
        self.num_slices = self.markov + 1



        super().__init__(num_samples)

    def sample_node(self, node_id, prev_data):
        node_name = self.index_to_node[node_id]
        link = self.links[node_name]
        ret = link(prev_data)
        return ret

    def children(self, id):
        row = self.adj_mat[id].squeeze()

        return row.nonzero()

    def parents(self, id):
        col = self.adj_mat[:,id].squeeze()
        return col.nonzero()

    @functools.lru_cache
    def execution_order(self):
        assert not check_symmetric(self.adj_mat)
        #assert not has_cycle(self.adj_mat)

        nodeset = set(torch.arange(self.num_nodes).tolist())

        column_sums = torch.sum(self.adj_mat, dim=0)
        # Get nodes with no parents
        nodes_no_parents = torch.where(column_sums == 0)[0]

        execution_order = nodes_no_parents.tolist()
        while set(execution_order) != nodeset:
            add_to_order = []
            for node in execution_order:
                for child in self.children(node):
                    if child in execution_order or child in add_to_order:
                        continue

                    childs_parents = self.parents(child)

                    all_in_execution_order = True
                    for childs_parent in childs_parents:
                        if childs_parent not in execution_order:
                            all_in_execution_order = False
                            break

                    if all_in_execution_order:
                        add_to_order.append(child.squeeze().item())
            execution_order = execution_order + add_to_order
        return execution_order

    def node_index_in_order(self, execution_order):
        dico = {}

        for i, node2 in enumerate(execution_order):
            dico[node2] = i

        return dico

    def resolve_interv(self, interv):
        # interv is set of ints (node ids) {i,j,k...}
        interv_names = {self.index_to_node[i] for i in interv}
        return interv_names

    def gen_one(self):
        def sample_node(node_id, prev_data):
            node_name = self.index_to_node[node_id]
            link = self.links[node_name]
            ret = link(prev_data)
            return ret

        def sample_node_interv(node_id):
            return self.unlinks[self.index_to_node[node_id]]()

        execution_order = self.execution_order()
        vec = torch.zeros(self.num_nodes)
        for node in execution_order:
            parents = self.parents(node)
            if len(parents) >= 0:
                parent_data = vec[parents]
            else:
                parent_data = None
            vec[node] = sample_node(node, parent_data)

        data = [vec] + [torch.clone(vec) for _ in range(self.markov)]
        #data = torch.concat(data)

        dict_node2execution = self.node_index_in_order(execution_order)

        interv_list = []
        interv_id_list = []

        intervention = None
        for m in range(1,self.markov+1,1):
            if intervention is None:
                intervention = self.intervset.init(1)
            else:
                intervention = self.intervset.pick(intervention)


            interv = intervention.self.squeeze()
            interv_id_list.append(interv.cpu().numpy())
            interv = self.intervset.id2interv(interv)

            interv_list.append(interv)
            if len(interv) == 0:
                continue

            # shortcut
            skip_to = dict_node2execution[interv[0]]    # first node in execution (others MUST be bigger)

            #interv_names = self.resolve_interv(interv)

            for node in execution_order[skip_to:]:
                if node in interv:
                    val = sample_node_interv(node)
                else:
                    val = sample_node(node, data[m-1])

                data[m][node] = val

        return data, interv_list, interv_id_list

    def generate(self):
        latents = []
        intervs = []
        interv_ids = []
        for _ in range(self.num_samples):
            lat, int, id = self.gen_one()


            interv_ids.append(np.array(id))
            latents.append(torch.vstack(lat))
            intervs.append(int)

        latents = torch.stack(latents)
        observations = latents * 2 # TODO replace this by affine transform
        interv_ids = np.stack(interv_ids)


        return latents, observations, intervs, torch.tensor(interv_ids), observations





        return z_i_s.detach(), ret.detach(), intervention.detach(), ret.detach()

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
        'A': lambda args: Normal(0.0, 1.0).sample(),
        'B': lambda args: Normal(0.3 * args[0] ** 2 - 0.6 * args[0], 0.8 ** 2).sample(),
        'C': lambda args: Normal(0.2 * args[0] ** 2 + -0.8 * args[1], 1.0).sample()
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

    switch_case = {
        0: Table(dict_of_tables, dict_of_alphas),
        #0: Uniform(no_replace=True),
        #2: Table(dict_of_tables, dict_of_alphas)
    }

    x.set_switch_case(switch_case)

    dataset = ToyNDDataset(100, G, links, unlinks, intervset=x)


    # To access a single sample
    sample = dataset[0]
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt


    def do_plot(data, interventions):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

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

        plot_data = data.reshape(len(dataset) * 2, 3)
        # how many intervs to push??????
        ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:,2])
        NUM_INTERVS_OF_EACH_TYPE_TO_PLOT = 2
        for i in interventions.unique():
            if i == 0:
                continue
            if i not in list(range(dataset.num_nodes)):
                continue

            # opt to select the first elements. doesn't change anything anyway.
            selected_intervs = torch.argsort(interventions == i, descending=True)
            selected_intervs = selected_intervs[:NUM_INTERVS_OF_EACH_TYPE_TO_PLOT].squeeze()

            sel_latents = data[selected_intervs]

            color = {
                1: "red",
                2: "blue",
                3: "green",
                0: None
            }
            plot_many_arrows(sel_latents, color=color[i.int().item()])
        plt.show()

    do_plot(dataset.latents, dataset.intervention_ids)
    #do_plot(dataset.observations, dataset.intervention_ids)
    exit()
