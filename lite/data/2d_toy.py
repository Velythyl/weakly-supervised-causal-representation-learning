import random
import torch
from torch.distributions import Normal
from torch.utils.data import Dataset

from utils import ConditionalAffineScalarTransform

class ToyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.transform = ConditionalAffineScalarTransform()
        self.data = self.generate()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def generate(self):


        
        z1 = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample_n(self.num_samples)
        if random.random() < 0.5:
            z2 = Normal(0.3 * z1 ** 2 - 0.6 * z1,  + 0.3, 0.8 * torch.eye(self.num_samples)).sample()
        else:
            z2 = Normal(torch.tensor([1.0]), torch.tensor([0.4])).sample_n(self.num_samples)

        x


def __main__():

    dataset = ToyDataset(10)

    # To access a single sample
    sample = dataset[0]
    print(sample)
