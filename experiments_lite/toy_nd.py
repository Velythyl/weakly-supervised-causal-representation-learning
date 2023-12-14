from typing import List

import torch
from torch import nn
from torch.distributions import Normal, Categorical, Dirichlet
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateFinder

import matplotlib.pyplot as plt
import seaborn as sns
import io

from einops import rearrange

from ws_crl_lite.datasets.toy_2d import Toy2dDataset 
from ws_crl_lite.models.ilcm import ILCMEncoder, ILCMDecoder
from ws_crl.transforms import make_mlp_structure_transform


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
        # ilcm_encoder,
        # ilcm_decoder,
        dim_x : int = 2,
        dim_z : int = 2,
        markov_len : int = 1,
        noise_encoder_hidden_dims : List[int] = [3, 3],
        noise_decoder_hidden_dims : List[int] = [3, 3],
        intervention_encoder_hidden_dims : List[int] = [3, 3],
        solution_function_hidden_layers : int = 2,
        solution_function_hidden_units : int = 64,
        solution_function_homoskedastic : bool = False,
        solution_function_min_std : float = 0.2,
        solution_function_init : str = "broad",
        solution_function_concat_masks_to_parents : bool = False,
        beta : float = 0.5,
        lr : float = 1e-4
    ):
        super().__init__()

        # optimisation parameters
        self.lr=lr
        self.beta = beta

        # settings
        self.markov_len = markov_len
        self.dim_z = dim_z
        self.dim_x = dim_x

        # NNs
        self.noise_encoder = self._make_mlp([dim_x] + noise_encoder_hidden_dims + [dim_z * 2])
        self.noise_decoder = self._make_mlp([dim_z] + noise_decoder_hidden_dims + [dim_x])
        # TODO: add constraints
        self.intervention_encoders = nn.ModuleList([
            self._make_mlp([dim_z] + intervention_encoder_hidden_dims + [(2 ** dim_z) ** 2]) for _ in range(markov_len)])
        self.alpha = nn.parameter.Parameter(torch.ones(self.markov_len))
        self.scm = nn.ModuleList([
            make_mlp_structure_transform(
                self.dim_z,
                hidden_layers=solution_function_hidden_layers,
                hidden_units=solution_function_hidden_units,
                homoskedastic=solution_function_homoskedastic,
                min_std=solution_function_min_std,
                initialization=solution_function_init,
                concat_masks_to_parents=solution_function_concat_masks_to_parents)
            ] for _ in range(dim_z))

        self.encoder = ilcm_encoder
        self.decoder = ilcm_decoder
        self.validation_step_outputs = []

    @staticmethod
    def _make_mlp(layer_dims):
        layers = [nn.Linear(layer_dims[0], layer_dims[1])]
        for i in range(1, len(layer_dims) - 1):
            layers.append(nn.ReLU)
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        return nn.Sequential(*layers)

    @staticmethod
    def binary(x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    
    @staticmethod
    def stoch_avg(t, dim=0):
        param = Dirichlet(torch.ones(t.shape[dim])).sample()
        param.device = t.device
        param.requires_grad = False
        return param @ t

    def training_step(self, batch, batch_idx):

        seq_len = batch.shape[1]
        x = rearrange(batch, 'b l x -> (b l) x')
        
        # noise posterior
        e_mean, e_logstd = self.noise_encoder(x).tensor_split(2, dim=-1)
        e_std = torch.exp(e_logstd)
        e_mean = rearrange(e_mean, '(b l) z) -> l b z', l=seq_len)
        e_std = rearrange(e_std, '(b l) z -> l b z', l=seq_len)

        # intervention posterior 
        first_order_models = []
        for k in range(self.markov_len):
            # TODO: unsure about this
            m = self.intervention_encoders[k](torch.abs(e_mean[0] - e_mean[k + 1]))
            first_order_models.append(m)

        pi = [nn.functional.one_hot(torch.zeros(1), num_classes=2 ** self.dim_z)]
        for t in range(1, seq_len):
            l = min(t, self.markov_len)
            p = torch.dot(                  # mixing the first order models
                nn.Softmax(self.alpha[:l]),             
                torch.tensor([pi[t - k].clone().detach() @ first_order_models[k] for k in range(l)])
                )
            pi.append(p)
        
        # sample interventions
        log_q_I, I = 0, [self.binary(torch.zeros(1), self.dim_z)]
        for t in range(1, seq_len):
            qI_t = Categorical(pi[t])
            I_t = qI_t.sample()
            I.append(self.binary(I_t, self.dim_z))
            log_q_I += qI_t.log_prob(I)
        
        I = torch.stack(I).T     #   dim_z, seq_len

        # sample noise
        log_q_e = 0
        e = torch.empty_like(e_mean)
        for j in I:
            samples = []
            for mean, std in zip(
                torch.tensor_split(e_mean[:, :, j], torch.nonzero(I[j])),
                torch.tensor_split(e_std[:, :, j], torch.nonzero(I[j]))
             ):
                mean_, std_ = self.stoch_avg(mean), self.stoch_avg(std)
                q = Normal(mean_, std_)
                e_sample = q.sample()
                log_q_e += q.log_prob(e_sample).sum()
                samples.append(e_sample * torch.ones_like(mean))
            e[:, :, j] = torch.stack(samples)

        # compute sample probs according to prior



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
