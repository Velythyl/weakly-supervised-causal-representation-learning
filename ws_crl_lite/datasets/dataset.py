import dataclasses
import random
import struct

import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


from repo.ws_crl.transforms import make_scalar_transform

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

        self.latents, self.observations, self.interventions, self.intervention_ids, self.data = self.generate()
        self.latents = maybe_detach(self.latents)
        self.observations = maybe_detach(self.observations)
        self.interventions = maybe_detach(self.interventions)
        # todo change this to a one-hot
        self.intervention_ids = maybe_detach(self.intervention_ids)
        self.data = maybe_detach(self.data)

        i=0


        """
        x i = self.observations[:,i,:]
        
        z i = self.latents[:,i,:]
        
        
        """

    @property
    def num_interv_types(self):
        return self.interventions.unique().shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

