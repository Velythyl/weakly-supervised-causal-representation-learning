import torch
from torch import nn
from torchvision.ops import MLP
from torch.distributions import Normal, OneHotCategorical

from lite.utils import ConditionalAffineScalarTransform


class ILCMEncoder(nn.Module):

    def __init__(self, noise_encoder, intervention_encoder):
        super().__init__()

        self.noise_encoder = noise_encoder
        self.intervention_encoder = intervention_encoder

    def stoch_avg(t1, t2, indices):
        param = torch.rand(t1[indices].shape, requires_grad=False)
        return param * t1[indices] + (1.0 - param) * t2[indices]
    
    def forward(self, x1, x2):
        
        e1_mean, e1_std = self.noise_encoder(x1)
        e2_mean, e2_std = self.noise_encoder(x2)
        
        intervention_probs = self.intervention_encoder(torch.abs(e1_mean - e2_mean))
        intervention_posterior = OneHotCategorical(intervention_probs)
        
        intervention = intervention_posterior.sample()[1:]
        log_q_I = intervention_posterior.log_prob(intervention)
        
        eps_mean, eps_std = e1_mean, e1_std
        eps_mean[1 - intervention] = self.stoch_avg(e1_mean, e2_mean, 1 - intervention)
        eps_std[1 - intervention] = self.stoch_avg(e1_std, e2_std, 1 - intervention)
        eps_posterior = Normal(eps_mean, eps_std)
        intervened_eps_posterior = Normal(e2_mean[intervention], e2_std[intervention])
        
        e1 = eps_posterior.sample()
        log_q_e1 = eps_posterior.log_prob(e1)

        e2 = e1
        e2[intervention] = intervened_eps_posterior.sample()
        log_q_e2 = intervened_eps_posterior.log_prob(e2[intervention])

        log_q = log_q_e1 + log_q_e2 + log_q_I

        return (e1, e2, intervention), log_q


class ILCMDecoder(nn.Module):

    def __init__(self, dim_z, noise_decoder):
        super().__init__()
        self.dim_z = dim_z
        self.noise_decoder = noise_decoder

        self.adjacency_matrix = nn.Parameter(torch.ones((dim_z, dim_z)), requires_grad=True)
        self.solution_fns = [ConditionalAffineScalarTransform(MLP()) for _ in range(dim_z)]


    def parents(inputs, idx, adjacency_matrix):
        mask = adjacency_matrix[idx]
        mask[idx] = 0
        return inputs * mask

    def forward(self, e1, e2, intervention):
        log_p_e1 = Normal(0, 1).log_prob(e1)
        log_p_I = -torch.log(self.dim_z + 1)

        z, logdet = self.solution_fns[intervention].inverse(
            inputs=e2[:, intervention:intervention+1], 
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

