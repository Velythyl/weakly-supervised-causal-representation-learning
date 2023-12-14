from typing import List

import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateFinder

import matplotlib.pyplot as plt
import seaborn as sns
import io

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
    def __init__(
        self,
        ilcm_encoder,
        ilcm_decoder,
        dim_z : int = 2,
        beta : float = 0.5,
        lr : float = 1e-4
    ):
        super().__init__()

        # optimisation parameters
        self.lr=lr
        self.beta = beta

        # settings
        self.dim_z = dim_z

        # # NNs
        self.encoder = ilcm_encoder
        self.decoder = ilcm_decoder

        # outputs
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):

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

    def validation_step(self, batch, batch_idx):
        x = batch[:, 0]
        e, _ = self.encoder.noise_encoder(x).tensor_split(2, dim=-1)
        noise = e.detach()
        A = torch.empty(self.dim_z, self.dim_z)
        noise.requires_grad=True
        torch.set_grad_enabled(True)
        for i in range(self.dim_z):
            context = noise.clone()
            context[i] = 0
            z, logdet = self.decoder.solution_fns[i].inverse(noise[:, i : i + 1], context)
            A[i] = (torch.autograd.grad(logdet, context, torch.ones_like(z), retain_graph=True)[0] ** 2).sum(0)
        
        self.validation_step_outputs.append(A)

    def on_validation_epoch_end(self):
        A = torch.stack(self.validation_step_outputs).mean(0)
        self.validation_step_outputs = []
        matrix = A.clone().detach()
        heatmap_img = self.create_heatmap(matrix)
        self.logger.experiment.add_image("Heatmap", heatmap_img, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def create_heatmap(self, matrix):
        fig, ax = plt.subplots()
        sns.heatmap(matrix, ax=ax, annot=True)
        plt.close(fig)
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
    model = ILCMLite(ilcm_encoder, ilcm_decoder, dim_z=dim_z)

    train_dataset = Toy2dDataset(10000)
    val_dataset = Toy2dDataset(5000)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader =  DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    trainer = L.Trainer(check_val_every_n_epoch=1, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
