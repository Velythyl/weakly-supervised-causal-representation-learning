import torch
from torch import nn
from torchvision.ops import MLP
from torch.distributions import Normal, OneHotCategorical

from ws_crl_lite.utils import ConditionalAffineScalarTransform


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

        self.adjacency_matrix = nn.Parameter(torch.zeros((dim_z, dim_z)), requires_grad=True)

        self.params_nets = nn.ModuleList([nn.Sequential(
                    nn.Linear(dim_z, 5), 
                    nn.ReLU(),
                    nn.Linear(5, 5),
                    nn.ReLU(),
                    nn.Linear(5, dim_z),
                ) for _ in range(self.dim_z)])

        self.solution_fns = [ConditionalAffineScalarTransform(self.params_nets[i]) for i in range(dim_z)]

    def parents(self, child_idx):
        a = torch.sigmoid(torch.triu(self.adjacency_matrix.clone(), diagonal=1))
        mask = torch.concat((a[:child_idx, child_idx], torch.zeros(1), (1 - a[child_idx, child_idx + 1:])))
        return mask

    def forward(self, e1, e2, intervention):
        log_p_e1 = Normal(0, 1).log_prob(e1).sum()
        log_p_I = -torch.log(torch.tensor(self.dim_z + 1)) * e1.shape[0]

        i_mask = intervention[:, 1:].bool()
        log_p_e2 = 0
        for i in range(self.dim_z):
            if i_mask[:, i].any():
                z, logdet = self.solution_fns[i].inverse(e2[:, i], context=self.parents(i) * e1)
                log_p_e2 += Normal(0, 1).log_prob(z).sum() + logdet.sum()

        log_p = log_p_e1 + log_p_e2 + log_p_I
        x1_hat, x2_hat = self.noise_decoder(e1), self.noise_decoder(e2)
        
        return (x1_hat, x2_hat), log_p

