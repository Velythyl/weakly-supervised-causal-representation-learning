import torch
from torch.distributions import Normal
from repo.lite.data.dataset import WSCRLDataset

from repo.ws_crl.transforms import make_scalar_transform

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


class ToyNDDataset(WSCRLDataset):
    def __init__(self, num_samples, adj_mat, dim):
        self.adj_mat = torch.from_numpy(adj_mat)
        self.num_nodes = adj_mat.shape[0]
        assert self.num_nodes == dim
        self.transform = make_scalar_transform(n_features=2, layers=5)    #ConditionalAffineScalarTransform()
        super().__init__(num_samples)

    def children(self, id):
        row = self.adj_mat[id].squeeze()

        return row.nonzero()

    def parents(self, id):
        col = self.adj_mat[:,id].squeeze()
        return col.nonzero()

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
    def generate(self):

        def base_dist(node_id, prevs, z_i_s_slice):
            # z_i_s_slice: [num_samples, num_nodes]

            if len(prevs) == 0:
                ret = Normal(0.0 + node_id, 1.0).sample_n(z_i_s_slice.shape[0]).squeeze()
                return ret

            assert not (z_i_s_slice[:,prevs] == 0).any()  # this basically says "the node's parents must have been filled before calling base_dist

            prev_mus = prevs.squeeze()

            mu = z_i_s_slice[:,prev_mus]
            if len(mu.shape) == 2:
                mu = torch.sum(mu, dim=1)
            mu += node_id/self.num_nodes * torch.ones((z_i_s_slice.shape[0],))

            ret = Normal(mu, 0.8 ** 2).sample_n(1).squeeze()
            return ret

        execution_order = self.execution_order()

        z_i_s = torch.zeros((self.num_samples, 2, self.num_nodes))
        for node in execution_order:
            z_i_s[:,0,node] = base_dist(node_id=node, prevs=self.parents(node), z_i_s_slice=z_i_s[:,0])

        z_i_s[:, 1, :] = torch.clone(z_i_s[:, 0, :])

        interventions = torch.randint(low=0, high=self.num_nodes+1, size=(self.num_samples,))  # 0 is always the empty intervention
        dict_node2execution = self.node_index_in_order(execution_order)

        for i, intervention in enumerate(interventions):
            intervention = intervention.item()
            if intervention == 0:
                continue

            shortcircuit_execution_order = execution_order[dict_node2execution[intervention-1]:]
            for node in shortcircuit_execution_order:

                if node == intervention-1:
                    prevs = torch.tensor([])
                else:
                    prevs = self.parents(node)

                z_i_s[i,1,node] = base_dist(node_id=node, prevs=prevs, z_i_s_slice=z_i_s[i,1][None])

        in_latents = z_i_s.reshape(self.num_samples * 2, self.num_nodes)
        observations, idk_wtf_this_is = self.transform.forward(in_latents)
        ret = observations.reshape(self.num_samples, 2, self.num_nodes).detach()

        return z_i_s, ret, interventions, ret

if __name__ == "__main__":
    import networkx as nx


    G = nx.DiGraph()

    # Add edges to the graph based on the structure you provided
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'C'), ('B', 'D')]
    G.add_edges_from(edges)
    adj_matrix = nx.adjacency_matrix(G)
    # Convert the adjacency matrix to a NumPy array (if needed)
    adj_array = adj_matrix.toarray()

    dataset = ToyNDDataset(1000, adj_array, 4)

    # To access a single sample
    sample = dataset[0]
    import matplotlib.pyplot as plt

    def do_plot(data, interventions):
        def plot_many_arrows(pairs, color):
            # pairs is of shape [n arrows, 2, (x,y)]

            base_x = pairs[:, 0, 0]
            base_y = pairs[:, 0, 1]

            end_x = pairs[:, 1, 0]
            end_y = pairs[:, 1, 1]
            dx = end_x - base_x
            dy = end_y - base_y
            for i in range(base_x.shape[0]):
                plt.arrow(base_x[i], base_y[i], dx[i], dy[i], color=color)

        plot_data = data.reshape(len(dataset) * 2, 2)
        # how many intervs to push??????
        plt.scatter(plot_data[:, 0], plot_data[:, 1])
        NUM_INTERVS_OF_EACH_TYPE_TO_PLOT = 2
        for i in interventions.unique():
            if i == 0:
                continue

            # opt to select the first elements. doesn't change anything anyway.
            selected_intervs = torch.argsort(interventions == i, descending=True)
            selected_intervs = selected_intervs[:NUM_INTERVS_OF_EACH_TYPE_TO_PLOT]

            sel_latents = data[selected_intervs]

            plot_many_arrows(sel_latents, color="blue" if i == 2 else "red")
        plt.show()

    do_plot(dataset.latents, dataset.interventions)
    do_plot(dataset.observations, dataset.interventions)
    exit()
