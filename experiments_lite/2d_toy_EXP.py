import torch
from torch import nn
from torch.distributions import Normal

import pytorch_lightning as L

from pytorch_lightning.callbacks import LearningRateFinder

import matplotlib.pyplot as plt
import seaborn as sns
import io
from torchvision.transforms.functional import to_tensor

from torch.utils.data import DataLoader
from ws_crl_lite.datasets.toy_2d import Toy2dDataset 

from ws_crl_lite.models.ilcm import ILCMEncoder, ILCMDecoder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class ILCMLite(L.LightningModule):
    def __init__(self, dim_z, ilcm_encoder, ilcm_decoder, beta=0.5, lr=1e-4):
        super().__init__()
        self.lr=lr
        self.dim_z = dim_z
        self.beta = beta
        self.encoder = ilcm_encoder
        self.decoder = ilcm_decoder

    def training_step(self, batch, batch_idx):
        batch_sz, seq_len, dim_z = batch.shape
        #TODO: make it work for sequences of longer than 2
        x1, x2 = batch[:, 0], batch[:, 1]
        (e1, e2, intervention), log_prob_posterior = self.encoder(x1, x2)
        (x1_hat, x2_hat), log_prob_prior = self.decoder(e1, e2, intervention)
        
        mse1 = Normal(x1_hat, 1).log_prob(x1).sum()
        mse2 = Normal(x2_hat, 1).log_prob(x2).sum()

        loss = (mse1 + mse2) + self.beta * (log_prob_prior - log_prob_posterior)
        self.log("reconst_loss", mse1 + mse2)
        self.log("kl_loss", log_prob_prior - log_prob_posterior)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_end(self):
        matrix = torch.sigmoid(self.decoder.adjacency_matrix.clone().detach())
        # Generate heatmap
        heatmap_img = self.create_heatmap(matrix)
        # Log heatmap
        self.logger.experiment.add_image("Heatmap", heatmap_img, self.current_epoch)

    def create_heatmap(self, matrix):
        fig, ax = plt.subplots()
        sns.heatmap(matrix, ax=ax, annot=True)
        plt.close(fig)  # Close the plot to prevent it from displaying in the notebook

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = to_tensor(plt.imread(buf))

        return image


if __name__ == '__main__':
    
    seq_len, dim_z, dim_x = 2, 2, 2 

    noise_encoder = nn.Sequential(nn.Linear(dim_x, 3), nn.ReLU(), nn.Linear(3, dim_z * 2))
    noise_decoder = nn.Sequential(nn.Linear(dim_z, 3), nn.ReLU(), nn.Linear(3, dim_x))
    intervention_encoder = nn.Sequential(nn.Linear(dim_z, 3), nn.ReLU(), nn.Linear(3, dim_z + 1), nn.Softmax())
    ilcm_encoder = ILCMEncoder(noise_encoder, intervention_encoder)   
    ilcm_decoder = ILCMDecoder(noise_decoder, dim_z)

    for name, param in ilcm_decoder.named_parameters():
        print(f"{name}: {param.size()}")


    model = ILCMLite(dim_z, ilcm_encoder, ilcm_decoder)
    dataset = Toy2dDataset(5000)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    trainer = L.Trainer(callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))])
    trainer.fit(model, train_dataloaders=train_loader)
