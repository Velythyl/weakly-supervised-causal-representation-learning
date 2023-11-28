import torch
from torch import nn
from torchvision.ops import MLP
from torch.distributions import Normal, OneHotCategorical, MultivariateNormal

from lite.utils import ConditionalAffineScalarTransform


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
        
        e1_mean, e1_logstd = self.noise_encoder(x1)
        e2_mean, e2_logstd = self.noise_encoder(x2)

        e1_std, e2_std = torch.exp(e1_logstd), torch.exp(e2_logstd)
        
        intervention_probs = self.intervention_encoder(torch.abs(e1_mean - e2_mean))
        intervention_posterior = OneHotCategorical(intervention_probs)
        
        intervention = intervention_posterior.sample()
        log_q_I = intervention_posterior.log_prob(intervention)

        i_mask = intervention[1:].bool()
        
        eps_mean, eps_std = e1_mean, e1_std
        unintervened_eps_mean = self.stoch_avg(e1_mean, e2_mean, ~i_mask)
        eps_mean = eps_mean.masked_scatter(~i_mask, unintervened_eps_mean)
        unintervened_eps_std = self.stoch_avg(e1_std, e2_std, ~i_mask)
        eps_std = eps_std.masked_scatter(~i_mask, unintervened_eps_std)

        eps_posterior = MultivariateNormal(eps_mean, torch.diag(eps_std))
        intervened_eps_posterior = MultivariateNormal(e2_mean[i_mask], torch.diag(e2_std[i_mask]))
        
        e1 = eps_posterior.sample()
        log_q_e1 = eps_posterior.log_prob(e1)

        e2 = e1
        log_q_e2 = 0
        if i_mask.any():
            e2[i_mask] = intervened_eps_posterior.sample()
            log_q_e2 += intervened_eps_posterior.log_prob(e2[i_mask])

        log_q = log_q_e1 + log_q_e2 + log_q_I

        return (e1, e2, intervention), log_q


class ILCMDecoder(nn.Module):

    def __init__(self, noise_decoder, dim_z):
        super().__init__()
        self.dim_z = dim_z
        self.noise_decoder = noise_decoder

        self.adjacency_matrix = nn.Parameter(torch.ones((dim_z, dim_z)), requires_grad=True)
        self.solution_fns = [ConditionalAffineScalarTransform(nn.Sequential(nn.Linear(dim_z, 3), nn.ReLU(), nn.Linear(3, dim_z))) for _ in range(dim_z)]

    @staticmethod
    def parents(inputs, idx, adjacency_matrix):
        mask = adjacency_matrix[idx]
        mask[idx] = 0
        return inputs * mask

    def forward(self, e1, e2, intervention):
        log_p_e1 = Normal(0, 1).log_prob(e1)
        log_p_I = -torch.log(torch.tensor(self.dim_z + 1))

        i_mask = intervention[1:].bool()

        z, logdet = self.solution_fns[i_mask].inverse(
            inputs=e2[i_mask], 
            context=self.parents(e1, intervention, self.adjacency_matrix)
            )
        log_p_e2 = (e1 - e2)**2
        log_p_e2[intervention] = Normal(0, 1).log_prob(z) + logdet
        log_p_e2 = log_p_e2.sum()

        log_p = log_p_e1 + log_p_e2 + log_p_I

        x1_hat, x2_hat = self.noise_decoder(e1), self.noise_decoder(e2)
        return (x1_hat, x2_hat), log_p



# class ILCMLite(nn.Module):

#     def __init__(self, dim_z):
#        super().__init__()
#        self.dim_z = dim_z

#        self.eps_prior = Normal(0, 1)
#        self.latent_prior = Normal(0, 1)
#        self.intervention_prior = OneHotCategorical(torch.ones(dim_z))

#        self.adjacency_matrix = nn.Parameter(torch.ones((dim_z, dim_z)), requires_grad=True)
#        self.solution_fns = [
#            ConditionalAffineScalarTransform(MLP()) for _ in range(dim_z)
#        ]

#        self.noise_encoder = MLP()
#        self.noise_decoder = MLP()
#        self.intervention_encoder = MLP()


#     def forward(self, x1, x2, beta=1.0):

#         (e1, e2, intervention), log_prob_posterior = self.encoder(x1, x2)
#         (x1_hat, x2_hat), log_prob_prior = self.decoder(e1, e2, intervention)
        
#         mse1 = torch.sum((x1_hat - x1) ** 2)
#         mse2 = torch.sum((x2_hat - x2) ** 2)

#         loss = mse1 + mse2 + beta * (log_prob_prior - log_prob_posterior)

#         return loss

#     def encoder(self, x1, x2):
        
#         e1_mean, e1_std = self.noise_encoder(x1)
#         e2_mean, e2_std = self.noise_encoder(x2)
        
#         intervention_probs = self.intervention_encoder(torch.abs(e1_mean - e2_mean))
#         intervention_posterior = OneHotCategorical(intervention_probs)
        
#         intervention = intervention_posterior.sample()
#         log_q_I = intervention_posterior.log_prob(intervention)

#         def stoch_avg(t1, t2, indices):
#             param = torch.rand(t1[indices].shape, requires_grad=False)
#             return param * t1[indices] + (1.0 - param) * t2[indices]
        
#         eps_mean, eps_std = e1_mean, e1_std
#         eps_mean[1 - intervention] = stoch_avg(e1_mean, e2_mean, 1 - intervention)
#         eps_std[1 - intervention] = stoch_avg(e1_std, e2_std, 1 - intervention)
#         eps_posterior = Normal(eps_mean, eps_std)
#         intervened_eps_posterior = Normal(e2_mean[intervention], e2_std[intervention])
        
#         e1 = eps_posterior.sample()
#         log_q_e1 = eps_posterior.log_prob(e1)

#         e2 = e1
#         e2[intervention] = intervened_eps_posterior.sample()
#         log_q_e2 = intervened_eps_posterior.log_prob(e2[intervention])

#         log_q = log_q_e1 + log_q_e2 + log_q_I

#         return (e1, e2, intervention), log_q


#     def decoder(self, e1, e2, intervention):
#         log_p_e1 = self.eps_prior.log_prob(e1)
#         log_p_I = self.intervention_prior.log_prob(intervention)

#         def parents(inputs, idx, adjacency_matrix):
#             mask = adjacency_matrix[idx]
#             mask[idx] = 0
#             return inputs * mask

#         z, logdet = self.solution_fns[intervention].inverse(
#             inputs=e2[:, intervention:intervention+1], 
#             context=parents(e1, intervention, self.adjacency_matrix)
#             )
#         log_p_e2 = (e1 - e2)**2
#         log_p_e2[intervention] = self.latent_prior.log_prob(z) + logdet
#         log_p_e2 = log_p_e2.sum()

#         log_p = log_p_e1 + log_p_e2 + log_p_I

#         x1_hat, x2_hat = self.noise_decoder(e1), self.noise_decoder(e2)
#         return (x1_hat, x2_hat), log_p

