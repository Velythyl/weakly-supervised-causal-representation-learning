import torch
import lightning as L


class ILCMLite(L.LightningModule):
    def __init__(self, dim_z, encoder, decoder, beta=1.0):
        super().__init__()
        self.dim_z = dim_z
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder

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


