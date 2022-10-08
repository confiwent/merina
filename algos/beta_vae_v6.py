"""
In this version, the auto encoder predicts the network throughputs only 
"""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class BetaVAE(torch.nn.Module):
    def __init__(self,
                 in_channels= 1,
                 hist_dim = 3,
                 hidden_size = 64,
                 layer_num_gru = 1,
                 latent_dim = 5,
                 hidden_dims = None,
                 beta = 0.4,
                 delta = 0.1,
                 gamma = 0.7):
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.FE_cnn_channels = 128
        self.hist_dim = hist_dim
        self.sequence_channels = in_channels
        self.kld_beta = beta
        self.kld_lambda = delta
        self.gamma = gamma
        self.prior_mu = torch.zeros(latent_dim).type(dtype)
        self.prior_logvar = torch.zeros(latent_dim).type(dtype)
        # self.loss_type = loss_type
        # self.C_max = torch.Tensor([max_capacity])
        # self.C_stop_iter = Capacity_max_iter

        # Build Encoder
        # self.encoder = nn.GRU(input_size = self.sequence_channels, hidden_size = hidden_size, num_layers = layer_num_gru, batch_first = True)

        self.FeatureExactor_CNN = nn.Sequential(
                                    nn.Conv1d(1, self.FE_cnn_channels, 4), # for available chunk sizes 6 version  L_out = 6 - (4-1) -1 + 1 = 3
                                    nn.LeakyReLU()
        )

        modules = []
        if hidden_dims is None:
            hidden_dims = [512]

        in_channels_ = (self.hist_dim - 3) * self.FE_cnn_channels * self.sequence_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels_, h_dim),
                    nn.LeakyReLU())
            )
            in_channels_ = h_dim

        self.mlp = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(in_channels_, self.latent_dim)
        self.fc_var = nn.Linear(in_channels_, self.latent_dim)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder 
        :return: (Tensor) List of latent codes
        """

        fut_inputs_1_ = input[:, :, -1]
        fut_inputs_1_ = fut_inputs_1_.view(-1, self.num_flat_features(fut_inputs_1_))
        fut_inputs_1_ = torch.unsqueeze(fut_inputs_1_, 1)

        fut_inputs_2_ = input[:, :, 0:1]
        fut_inputs_2_ = fut_inputs_2_.view(-1, self.num_flat_features(fut_inputs_2_))
        fut_inputs_2_ = torch.unsqueeze(fut_inputs_2_, 1)

        FE_cnn_1 = self.FeatureExactor_CNN(fut_inputs_1_)
        FE_cnn_1_ = FE_cnn_1.view(-1, self.num_flat_features(FE_cnn_1))

        FE_cnn_2 = self.FeatureExactor_CNN(fut_inputs_2_)
        FE_cnn_2_ = FE_cnn_1.view(-1, self.num_flat_features(FE_cnn_2))

        hidden_inputs = torch.cat((FE_cnn_1_, FE_cnn_2_), dim = 1)

        result = self.mlp(hidden_inputs)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var # [batch, latent_dim]

    def get_latent(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    # def decode(self, z, his_input):
    #     his_input_ = his_input.view(-1, self.num_flat_features(his_input))
    #     input = torch.cat((his_input_, z), dim = 1)
    #     result = self.decoder_input(input)
    #     result = self.decoder(result)
    #     result_mu = self.final_layer_mu(result)
    #     result_sigma = self.final_layer_sigma(result)
    #     return result_mu, result_sigma

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def log_gaussian_loss(self, output, target, sigma):
        output = torch.exp(output)
        exponent = 0.5 * (target - output)**2 / sigma**2
        log_coeff = torch.log(sigma) + 0.5*np.log(2*np.pi) # or torch.log(torch.from_numpy(np.pi))
        return torch.mean(exponent + log_coeff, dim = 0)
        
        # if sum_reduce:
        #     return -(log_coeff + exponent).sum()
        # else:
        #     return -(log_coeff + exponent)

    def forward(self, trajectory_input):
        mu, log_var = self.encode(trajectory_input)
        return mu, log_var

    def loss_function(self, mu, log_var):
        # num_batch = mu.size()[0] # Get the batch size 
        # prior_mu = self.prior_mu.repeat(num_batch, 1)
        # prior_log_var = self.prior_logvar.repeat(num_batch, 1)

        # kld_loss_latent = torch.mean(-0.5 * torch.sum(1 + log_var - prior_log_var - (mu - prior_mu) ** 2 / prior_log_var.exp() - (log_var - prior_log_var).exp(), dim = 1), dim = 0)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # beta vae loss
        # reg_loss = self.kld_lambda * kld_loss_reg
        kld_loss = self.kld_beta * kld_loss

        return kld_loss


    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

    def update_prior(self, mu_new, log_var_new):
        self.prior_mu = mu_new
        self.prior_logvar = log_var_new

    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

    # def generate(self, x: Tensor, **kwargs) -> Tensor:
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """

    #     return self.forward(x)[0]