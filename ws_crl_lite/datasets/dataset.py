import dataclasses
import random
import struct

import torch
from torch.distributions import Normal
from torch.utils.data import Dataset



def maybe_detach(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach()
    return arr

@dataclasses.dataclass
class WSCRLData:
    latents: torch.Tensor
    observations: torch.Tensor
#self.interventions = maybe_detach(self.interventions)
## todo change this to a one-hot
#self.intervention_ids = maybe_detach(self.intervention_ids)


class WSCRLDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.latents, self.observations, self.interventions, self.intervention_ids = self.generate()
        self.latents = maybe_detach(self.latents)
        self.observations = maybe_detach(self.observations)
        self.interventions = maybe_detach(self.interventions)
        # todo change this to a one-hot
        self.intervention_ids = maybe_detach(self.intervention_ids)


        """
        x i = self.observations[:,i,:]
        
        z i = self.latents[:,i,:]
        
        
        """

    @property
    def num_interv_types(self):
        return self.intervention_ids.unique().shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.observations[idx]

