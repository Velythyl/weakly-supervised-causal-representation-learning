import torch
from torch import nn
from torchvision.ops import MLP
from torch.distributions import Normal, OneHotCategorical


from ws_crl.utils import clean_and_clamp
from ws_crl.transforms import make_mlp_structure_transform

class ILCMEncoder(nn.Module):

    def __init__(self, noise_encoder, intervention_encoder):
        super().__init__()

        self.noise_encoder = noise_encoder
        self.intervention_encoder = intervention_encoder
    
    @staticmethod
    def stoch_avg(t1, t2, mask):
        param = torch.rand(t1[mask].shape, device=t1.device, requires_grad=False)
        return param * t1[mask] + (1.0 - param) * t2[mask]
    
    def forward(self, x1, x2):
        
        e1_mean, e1_logstd = self.noise_encoder(x1).tensor_split(2, dim=-1)
        e2_mean, e2_logstd = self.noise_encoder(x2).tensor_split(2, dim=-1)

        e1_std, e2_std = torch.exp(e1_logstd), torch.exp(e2_logstd)
        
        intervention_probs = self.intervention_encoder(torch.abs(e1_mean - e2_mean))
        intervention_posterior = OneHotCategorical(intervention_probs)
        
        intervention = intervention_posterior.sample()
        log_q_I = intervention_posterior.log_prob(intervention).sum()

        i_mask = intervention[:, 1:].bool()
        
        eps_mean, eps_std = e1_mean, e1_std
        unintervened_eps_mean = self.stoch_avg(e1_mean, e2_mean, ~i_mask)
        eps_mean = eps_mean.masked_scatter(~i_mask, unintervened_eps_mean)
        unintervened_eps_std = self.stoch_avg(e1_std, e2_std, ~i_mask)
        eps_std = eps_std.masked_scatter(~i_mask, unintervened_eps_std)

        eps_posterior = Normal(eps_mean, eps_std)
        intervened_eps_posterior = Normal(e2_mean[i_mask], e2_std[i_mask])
        
        e1 = eps_posterior.sample()
        log_q_e1 = eps_posterior.log_prob(e1).sum()

        e2 = e1
        log_q_e2 = 0
        if i_mask.any():
            e2[i_mask] = intervened_eps_posterior.sample()
            log_q_e2 += intervened_eps_posterior.log_prob(e2[i_mask]).sum()

        log_q = log_q_e1 + log_q_e2 + log_q_I

        return (e1, e2, intervention), log_q


class ILCMDecoder(nn.Module):

    def __init__(self, noise_decoder, dim_z):
        super().__init__()
        self.dim_z = dim_z
        self.noise_decoder = noise_decoder
        self.solution_fns = nn.ModuleList([make_mlp_structure_transform(
                    self.dim_z,
                    hidden_layers=2,
                    hidden_units=64,
                    homoskedastic=False,
                    min_std=0.2,
                    initialization="broad",
                    concat_masks_to_parents=False
                ) for _ in range(self.dim_z)])

    def forward(self, e1, e2, intervention):
        log_p_e1 = clean_and_clamp(Normal(0, 1).log_prob(e1)).sum()
        log_p_I = -torch.log(torch.tensor(self.dim_z + 1)) * e1.shape[0]

        i_mask = intervention[:, 1:]
        log_p_e2 = 0
        for i in range(self.dim_z):
            if i_mask[:, i].any():
                context = e1.clone()
                context[:, i] = 0
                z, logdet = self.solution_fns[i].inverse(e2[:, i : i + 1], context)
                log_p_e2 += ((Normal(0, 1).log_prob(clean_and_clamp(z)) + logdet) * i_mask[:, i]).sum()

        log_p = log_p_e1 + log_p_e2 + log_p_I
        x1_hat, x2_hat = self.noise_decoder(e1), self.noise_decoder(e2)
        
        return (x1_hat, x2_hat), log_p
