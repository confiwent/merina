import torch
import torch.nn as nn
import torch.nn.functional as F


FE_CNN_unum = 128
FE_MLP_unum = 128
FE_latent_unum = 1024

class Actor(torch.nn.Module):
    def __init__(self, a_dim, latent_dim, state_channels, his_dim, hidden_layers = None):
        super(Actor, self).__init__()
        self.state_channels = state_channels
        self.a_dim = a_dim
        self.latent_dim = latent_dim
        self.his_dim = his_dim
        self.hidden_layers = hidden_layers
        self.FE_cnn_channels = FE_CNN_unum
        self.FE_output = FE_MLP_unum
        self.latent_mlp_output = FE_latent_unum
        self.FE_mlp_input = 1

        if self.hidden_layers is None:
            self.hidden_layers = [128]

        self.FeatureExactor_MLP_1 = nn.Sequential(
                                    nn.Linear(self.FE_mlp_input, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_MLP_2 = nn.Sequential(
                                    nn.Linear(self.FE_mlp_input, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_MLP_3 = nn.Sequential(
                                    nn.Linear(self.FE_mlp_input, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_TD_1 = nn.Sequential(
                                    nn.Linear(self.his_dim, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_TD_2 = nn.Sequential(
                                    nn.Linear(self.his_dim, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_CNN = nn.Sequential(
                                    nn.Conv1d(1, self.FE_cnn_channels, 4), # for available chunk sizes 6 version  L_out = 6 - (4-1) -1 + 1 = 3
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_latent = nn.Sequential(
                                    nn.Linear(self.latent_dim, self.latent_mlp_output),
                                    nn.LeakyReLU()
        )

        modules = []

        in_channels_ = self.latent_mlp_output + 3 * self.FE_cnn_channels + 5 * self.FE_output
        for h_dim in self.hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels_, h_dim),
                    nn.LeakyReLU())
            )
            in_channels_ = h_dim

        #===================Hidden layer=========================
        self.mlp = nn.Sequential(*modules)

        #==================Output Layers======================
        self.policy = nn.Sequential(
                        nn.Linear(in_channels_, self.a_dim),
                        nn.Softmax(dim=1))

    # @profile
    def forward(self, states_input, latent_input):
        ## inital inputs preparation
        # past throughput
        thp_info_ = states_input[:, 0:1, :]
        thp_info_ = thp_info_.view(-1, self.num_flat_features(thp_info_))

        # video chunk sizes 
        fut_inputs_1_ = states_input[:, 4:10, -1]
        fut_inputs_1_ = fut_inputs_1_.view(-1, self.num_flat_features(fut_inputs_1_))
        fut_inputs_1_ = torch.unsqueeze(fut_inputs_1_, 1) ## (batch, 1, 6)
        
        # current bufffer size 
        buffer_size_ = states_input[:, 1:2, -1]
        buffer_size_ = buffer_size_.view(-1, self.num_flat_features(buffer_size_))

        # last_bitrate version
        quality_level_ = states_input[:, 2:3, -1]
        quality_level_ = quality_level_.view(-1, self.num_flat_features(quality_level_))

        # video chunks remain
        chunk_remain_ = states_input[:, 3:4, -1]
        chunk_remain_ = chunk_remain_.view(-1, self.num_flat_features(chunk_remain_))

        # past download time 
        download_time_ = states_input[:, 10:11, :]
        download_time_ = download_time_.view(-1, self.num_flat_features(download_time_))

        ## process feature exactor 
        # video chunk sizes 384
        FE_cnn_1 = self.FeatureExactor_CNN(fut_inputs_1_)
        FE_cnn_1 = FE_cnn_1.view(-1, self.num_flat_features(FE_cnn_1))

        # latent information (sampled) 1280
        FE_mlp_latent_ = self.FeatureExactor_latent(latent_input)
        FE_mlp_latent_ = FE_mlp_latent_.view(-1, self.num_flat_features(FE_mlp_latent_))

        # remain infos (128 for each)
        FE_mlp_info_1 = self.FeatureExactor_MLP_1(buffer_size_)
        FE_mlp_info_1 =  FE_mlp_info_1.view(-1, self.num_flat_features(FE_mlp_info_1))

        FE_mlp_info_2 = self.FeatureExactor_MLP_2(quality_level_)
        FE_mlp_info_2 =  FE_mlp_info_2.view(-1, self.num_flat_features(FE_mlp_info_2))

        FE_mlp_info_3 = self.FeatureExactor_MLP_3(chunk_remain_)
        FE_mlp_info_3 =  FE_mlp_info_3.view(-1, self.num_flat_features(FE_mlp_info_3))

        FE_mlp_info_t = self.FeatureExactor_TD_1(thp_info_)
        FE_mlp_info_t = FE_mlp_info_t.view(-1, self.num_flat_features(FE_mlp_info_t))

        FE_mlp_info_d = self.FeatureExactor_TD_2(download_time_)
        FE_mlp_info_d = FE_mlp_info_d.view(-1, self.num_flat_features(FE_mlp_info_d))

        ## Aggregation

        hidden_inputs = torch.cat((FE_cnn_1, FE_mlp_latent_, FE_mlp_info_1,\
                                     FE_mlp_info_2, FE_mlp_info_3, FE_mlp_info_t, \
                                        FE_mlp_info_d), dim = 1)
        
        hiddens = self.mlp(hidden_inputs)

        actor_output = self.policy(hiddens)
        return actor_output

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

class Critic(torch.nn.Module):
    def __init__(self, a_dim, latent_dim, state_channels, his_dim, hidden_layers = None):
        super(Critic, self).__init__()
        self.state_channels = state_channels
        self.a_dim = a_dim
        self.latent_dim = latent_dim
        self.his_dim = his_dim
        self.hidden_layers = hidden_layers
        self.FE_cnn_channels = FE_CNN_unum
        self.FE_output = FE_MLP_unum
        self.latent_mlp_output = FE_latent_unum
        self.FE_mlp_input = 1

        if self.hidden_layers is None:
            self.hidden_layers = [128]

        self.FeatureExactor_MLP_1 = nn.Sequential(
                                    nn.Linear(self.FE_mlp_input, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_MLP_2 = nn.Sequential(
                                    nn.Linear(self.FE_mlp_input, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_MLP_3 = nn.Sequential(
                                    nn.Linear(self.FE_mlp_input, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_TD_1 = nn.Sequential(
                                    nn.Linear(self.his_dim, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_TD_2 = nn.Sequential(
                                    nn.Linear(self.his_dim, self.FE_output),
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_CNN = nn.Sequential(
                                    nn.Conv1d(1, self.FE_cnn_channels, 4), # for available chunk sizes 6 version  L_out = 6 - (4-1) -1 + 1 = 3
                                    nn.LeakyReLU()
        )

        self.FeatureExactor_latent = nn.Sequential(
                                    nn.Linear(self.latent_dim, self.latent_mlp_output),
                                    nn.LeakyReLU()
        )

        modules = []

        in_channels_ = self.latent_mlp_output + 3 * self.FE_cnn_channels + 5 * self.FE_output
        for h_dim in self.hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels_, h_dim),
                    nn.LeakyReLU())
            )
            in_channels_ = h_dim

        #===================Hidden layer=========================
        self.mlp = nn.Sequential(*modules)

        #==================Output Layers======================
        self.value = nn.Linear(in_channels_, 1)

    def forward(self, states_input, latent_input):
        ## inital inputs preparation
        # past throughput
        thp_info_ = states_input[:, 0:1, :]
        thp_info_ = thp_info_.view(-1, self.num_flat_features(thp_info_))

        # video chunk sizes 
        fut_inputs_1_ = states_input[:, 4:10, -1]
        fut_inputs_1_ = fut_inputs_1_.view(-1, self.num_flat_features(fut_inputs_1_))
        fut_inputs_1_ = torch.unsqueeze(fut_inputs_1_, 1)
        
        # current bufffer size 
        buffer_size_ = states_input[:, 1:2, -1]
        buffer_size_ = buffer_size_.view(-1, self.num_flat_features(buffer_size_))

        # last_bitrate version
        quality_level_ = states_input[:, 2:3, -1]
        quality_level_ = quality_level_.view(-1, self.num_flat_features(quality_level_))

        # video chunks remain
        chunk_remain_ = states_input[:, 3:4, -1]
        chunk_remain_ = chunk_remain_.view(-1, self.num_flat_features(chunk_remain_))

        # past download time 
        download_time_ = states_input[:, 10:11, :]
        download_time_ = download_time_.view(-1, self.num_flat_features(download_time_))

        ## process feature exactor 
        # video chunk sizes 384
        FE_cnn_1 = self.FeatureExactor_CNN(fut_inputs_1_)
        FE_cnn_1 = FE_cnn_1.view(-1, self.num_flat_features(FE_cnn_1))

        # latent information (sampled) 1280
        FE_mlp_latent_ = self.FeatureExactor_latent(latent_input)
        FE_mlp_latent_ = FE_mlp_latent_.view(-1, self.num_flat_features(FE_mlp_latent_))

        # remain infos (128 for each)
        FE_mlp_info_1 = self.FeatureExactor_MLP_1(buffer_size_)
        FE_mlp_info_1 =  FE_mlp_info_1.view(-1, self.num_flat_features(FE_mlp_info_1))

        FE_mlp_info_2 = self.FeatureExactor_MLP_2(quality_level_)
        FE_mlp_info_2 =  FE_mlp_info_2.view(-1, self.num_flat_features(FE_mlp_info_2))

        FE_mlp_info_3 = self.FeatureExactor_MLP_3(chunk_remain_)
        FE_mlp_info_3 =  FE_mlp_info_3.view(-1, self.num_flat_features(FE_mlp_info_3))

        FE_mlp_info_t = self.FeatureExactor_TD_1(thp_info_)
        FE_mlp_info_t = FE_mlp_info_t.view(-1, self.num_flat_features(FE_mlp_info_t))

        FE_mlp_info_d = self.FeatureExactor_TD_2(download_time_)
        FE_mlp_info_d = FE_mlp_info_d.view(-1, self.num_flat_features(FE_mlp_info_d))

        ## Aggregation

        hidden_inputs = torch.cat((FE_cnn_1, FE_mlp_latent_, FE_mlp_info_1,\
                                     FE_mlp_info_2, FE_mlp_info_3, FE_mlp_info_t, \
                                        FE_mlp_info_d), dim = 1)
        
        hiddens = self.mlp(hidden_inputs)

        critic_output = self.value(hiddens)
        return critic_output

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features
