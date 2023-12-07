import random
import struct

import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


from repo.ws_crl.transforms import make_scalar_transform

class WSCRLDataset(Dataset):
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

