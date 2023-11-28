import random
import struct

import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


from repo.ws_crl.transforms import make_scalar_transform

class ToyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.transform = make_scalar_transform(n_features=2, layers=5)    #ConditionalAffineScalarTransform()

        self.latents, self.observations, self.interventions, self.data = self.generate()

    @property
    def num_interv_types(self):
        return self.interventions.unique().shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def generate(self):
        z1 = Normal(0.0, 1.0).sample_n(self.num_samples).squeeze()
        z1_z1interv = Normal(0.0, 1.0).sample_n(self.num_samples).squeeze()
        z2 = Normal((0.3 * z1 ** 2 - 0.6 * z1).squeeze(), 0.8 ** 2).sample_n(1).squeeze()
        z2_z1interv = Normal((0.3 * z1_z1interv ** 2 - 0.6 * z1_z1interv).squeeze(), 0.8 ** 2).sample_n(1).squeeze()
        z2_z2interv = Normal(1.0, 0.4).sample_n(self.num_samples).squeeze()

        latents = torch.zeros((self.num_samples, 2, 2))

        latents[:,0,0] = z1
        latents[:,0,1] = z2

        intervention_ids = torch.randint(low=0,high=3,size=(self.num_samples,))

        second_z1 = torch.zeros_like(z1)
        second_z1[intervention_ids == 0] = z1[intervention_ids == 0]    # empty
        second_z1[intervention_ids == 1] = z1_z1interv[intervention_ids == 1] # interv on z1
        second_z1[intervention_ids == 2] = z1[intervention_ids == 2]    # interv on z2

        latents[:,1,0] = second_z1

        second_z2 = torch.zeros_like(z2)
        second_z2[intervention_ids == 0] = z2[intervention_ids == 0]    # empty
        second_z2[intervention_ids == 1] = z2_z1interv[intervention_ids == 1]   # interv on z1
        second_z2[intervention_ids == 2] = z2_z2interv[intervention_ids == 2]   # interv on z2

        latents[:,1,1] = second_z2.detach()

        in_latents = latents.reshape(self.num_samples * 2, 2)

        observations, idk_wtf_this_is = self.transform.forward(in_latents)
        ret = observations.reshape(self.num_samples, 2, 2).detach()
        return latents, ret, intervention_ids, ret

if __name__ == "__main__":

    dataset = ToyDataset(1000)

    # To access a single sample
    sample = dataset[0]

    import matplotlib.pyplot as plt
    import numpy as np
    latents = dataset.latents
    obs = dataset.observations
    print(dataset.observations)

    plot_lat = latents.reshape(len(dataset) * 2, 2)
    plt.scatter(plot_lat[:,0], plot_lat[:,1])
    plt.show()

    plot_lat = obs.reshape(len(dataset) * 2, 2)
    plt.scatter(plot_lat[:,0], plot_lat[:,1])
    plt.show()

    def plot_many_arrows(pairs, color):
        # pairs is of shape [n arrows, 2, (x,y)]

        base_x = pairs[:,0,0]
        base_y = pairs[:,0,1]

        end_x = pairs[:,1,0]
        end_y = pairs[:,1,1]
        dx = end_x - base_x
        dy = end_y - base_y
        for i in range(base_x.shape[0]):
            plt.arrow(base_x[i], base_y[i], dx[i], dy[i], color=color)


    # how many intervs to push??????
    plot_lat = dataset.latents.reshape(len(dataset) * 2, 2)
    plt.scatter(plot_lat[:, 0], plot_lat[:, 1])
    NUM_INTERVS_OF_EACH_TYPE_TO_PUSH = 2
    for i in range(1,dataset.num_interv_types):
        # opt to select the first elements. Doesn't change anything anyway.
        selected_intervs = torch.argsort(dataset.interventions == i, descending=True)
        selected_intervs = selected_intervs[:NUM_INTERVS_OF_EACH_TYPE_TO_PUSH]

        sel_latents = latents[selected_intervs]
        sel_obs = obs[selected_intervs]

        plot_many_arrows(sel_latents, color="blue" if i == 2 else "red")
    plt.show()



    exit()

    print(sample)
