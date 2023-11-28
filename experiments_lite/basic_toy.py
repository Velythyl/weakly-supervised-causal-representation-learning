import torch
from torch import nn
import pytorch_lightning as L

from ws_crl_lite.models.ilcm import ILCMEncoder, ILCMDecoder


class ILCMLite(L.LightningModule):
    def __init__(self, dim_z, ilcm_encoder, ilcm_decoder, beta=1.0):
        super().__init__()
        self.dim_z = dim_z
        self.beta = beta
        self.encoder = ilcm_encoder
        self.decoder = ilcm_decoder

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        (e1, e2, intervention), log_prob_posterior = self.encoder(x1, x2)
        (x1_hat, x2_hat), log_prob_prior = self.decoder(e1, e2, intervention)

        mse1 = torch.sum((x1_hat - x1) ** 2)
        mse2 = torch.sum((x2_hat - x2) ** 2)

        loss = (mse1 + mse2) + self.beta * (log_prob_prior - log_prob_posterior)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dim_z, dim_x = 2, 2

    noise_encoder = nn.Sequential(nn.Linear(dim_x, 3), nn.ReLU(), nn.Linear(3, 2))
    noise_decoder = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, dim_x))
    intervention_encoder = nn.Sequential(
        nn.Linear(dim_z, 3), nn.ReLU(), nn.Linear(3, dim_z + 1), nn.Softmax()
    )
    ilcm_encoder = ILCMEncoder(noise_encoder, intervention_encoder)
    ilcm_decoder = ILCMDecoder(noise_decoder, dim_z)

    model = ILCMLite(dim_z, ilcm_encoder, ilcm_decoder)

    import torch
    from torch.utils.data import Dataset, DataLoader

    # Define a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Generate a random 2x2 tensor
            tensor = torch.rand(2, 2)
            return tensor

    dataset = CustomDataset(5)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # for x1, x2 in train_loader:
    #     (e1, e2, intervention), log_prob_posterior = ilcm_encoder(x1, x2)
    #     print((e1, e2, intervention), log_prob_posterior)

    trainer = L.Trainer()
    trainer.fit(model, train_dataloaders=train_loader)
