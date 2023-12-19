import torch
from torch.distributions import Normal

from ws_crl_lite.datasets.dataset import WSCRLDataset, n_node_dataset
from ws_crl_lite.datasets.intervset import IntervTable, IntervSet

if __name__ == "__main__":
    import networkx as nx

    # FIRST, CREATE A GRAPH
    G = nx.DiGraph()
    # Add edges to the graph
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    G.add_edges_from(edges)

    AUTO = True
    if AUTO:
        dataset = n_node_dataset(1, 3, num_samples=50, timesteps=2, markov=2)[0]  # AutomaticDataset(1000, 6, 2, G)
    else:
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
        alpha_vec = np.random.uniform(0.1,1, size=(3,))
        # fixme
        # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
        switch_case = IntervTable(dict_of_tables, alpha_vec)

        x.set_tables(switch_case)
        x.kill(intervs_of_size=2, intervs_in_set={1:[3], 2: [1]})
        x.kill(nodes_in_intervs=[(1,2)])
        temp = x.impossible_intervention_ids

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


    plot_3d(dataset.latents)
    plot_3d(dataset.observations)
    # do_plot(dataset.latents, dataset.intervention_ids.squeeze(), "black")
    # do_plot(dataset.observations, dataset.intervention_ids)
    exit()