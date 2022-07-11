"""design for IMRL"""

import os
# from numba import jit
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import logging
from torch.utils.tensorboard import SummaryWriter

# from algos.mpc_pruning import A_DIM
# # from model_AQ import Actor, Critic
# from .agent_il_v4 import IML_agent
from .AC_net_v6 import Actor
from .beta_vae_v6 import BetaVAE
from .MPC_expert_v6 import ABRExpert
from .test_v5 import valid
from .replay_memory import ReplayMemory

RANDOM_SEED = 28
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
LEARNING_RATE_VAE = 1e-4
REWARD_MAX = 4.3
MAX_GRAD_NORM = 5.
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
DB_NORM_FACTOR = 25.0
DEFAULT_QUALITY = int(1)  # default video quality without agent
UPDATE_PER_EPOCH = 30 # update the parameters 8 times per epoch
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor

def train_iml_v4(train_epoch, net_env, valid_env, args, add_str, summary_dir):
    log_file_name = summary_dir + '/' + add_str + '/log'
    if not os.path.exists(summary_dir + '/' + add_str):
        os.mkdir(summary_dir + '/' + add_str)
    command = 'rm ' + summary_dir + '/' + add_str + '/*'
    os.system(command)
    # logging.basicConfig(filename=log_file_name + '_central',
    #                     filemode='w',
    #                     level=logging.INFO)
    writer = SummaryWriter(summary_dir + '/' + add_str + '/')

    s_info, s_len, c_len, \
        total_chunk_num, bitrate_versions, \
            rebuffer_penalty, smooth_penalty = net_env.get_env_info()

    latent_dim, mpc_horizon, gamma = args.latent_dim, args.mpc_h, args.gae_gamma
    kld_beta, kld_lambda, recon_gamma = args.kld_beta, args.kld_lambda, args.vae_gamma
    coeff_alpha, coeff_beta, coeff_gamma = args.lc_alpha, args.lc_beta, args.lc_gamma
    
    with open(log_file_name + '_record', 'w') as log_file, \
        open(log_file_name + '_test', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        br_dim = len(bitrate_versions)

        # Initialize the beta vae module
        vae_in_channels = 2 #1 + 2 * br_dim
        vae_net = BetaVAE(in_channels=vae_in_channels, hist_dim=c_len, \
                            latent_dim=latent_dim, beta=kld_beta, \
                                delta = kld_lambda, gamma =recon_gamma).type(dtype)
        optimiser_vae = torch.optim.Adam(vae_net.parameters(), lr= LEARNING_RATE_VAE)

        # Initialize the rl agent
        # imitation learning with a latent-conditioned policy
        model_actor = Actor(br_dim, latent_dim, s_info, s_len).type(dtype) 
        optimizer_actor = optim.Adam(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)

        # Initialize the expert agent
        expert = ABRExpert(net_env, rebuffer_penalty, smooth_penalty, \
                            int(mpc_horizon), int(total_chunk_num), bitrate_versions)

        # --------------------- Interaction with environments -----------------------

        # define the observations for vae
        # observations for vae input
        ob = np.zeros((vae_in_channels, c_len)) 
        state = np.zeros((s_info, s_len)) # define the state for rl agent
        # state = torch.from_numpy(state)
        
        bit_rate_ = bit_rate_opt = last_bit_rate = DEFAULT_QUALITY
        time_stamp, end_flag, video_chunk_remain = 0., True, total_chunk_num

        # define the replay memory
        steps_in_episode, minibatch_size = args.ro_len, args.batch_size
        epoch = 0
        memory = ReplayMemory(320 * steps_in_episode)
        
        for _ in range(train_epoch):
            # exploration in environments with expert strategy
            # memory.clear()
            model_actor.eval()
            vae_net.eval()
            states = []
            tar_actions = []
            obs = []
            rewards = []
            for _ in range(steps_in_episode):
                # record the current state, observation and action
                if not end_flag:
                    states.append(state_)
                    obs.append(ob_)
                    tar_actions.append(
                            torch.from_numpy(np.array([bit_rate_opt])).type(dlongtype)
                            )

                # behavior policy, both use expert's trajectories and randomly trajectories 
                
                bit_rate = bit_rate_

                # execute a step forward
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                        end_of_video, video_chunk_remain, _ = expert.step(bit_rate)

                # ----compute and record the reward of current chunk ------
                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                if args.log:
                    log_bit_rate = np.log(bitrate_versions[bit_rate] / \
                                            float(bitrate_versions[0]))
                    log_last_bit_rate = np.log(bitrate_versions[last_bit_rate] / \
                                                float(bitrate_versions[0]))
                    reward = log_bit_rate \
                            - rebuffer_penalty * rebuf \
                            - smooth_penalty * np.abs(log_bit_rate - log_last_bit_rate)
                else:
                    reward = bitrate_versions[bit_rate] / M_IN_K \
                            - rebuffer_penalty * rebuf \
                            - smooth_penalty * np.abs(bitrate_versions[bit_rate] -
                                                    bitrate_versions[last_bit_rate]) / M_IN_K
                # rewards.append(float(reward/REWARD_MAX))
                reward_max = rebuffer_penalty
                r_ = float(max(reward, -0.5*reward_max) / reward_max)
                rewards.append(r_)
                last_bit_rate = bit_rate

                # -------------- logging -----------------
                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                            str(bitrate_versions[bit_rate]) + '\t' +
                            str(bitrate_versions[bit_rate_opt]) + '\t' +
                            str(buffer_size) + '\t' +
                            str(rebuf) + '\t' +
                            str(video_chunk_size) + '\t' +
                            str(delay) + '\t' +
                            str(reward) + '\n')
                log_file.flush()

                ## dequeue history record
                state = np.roll(state, -1, axis=1)
                ob = np.roll(ob, -1, axis=1)

                # this should be S_INFO number of terms
                # state[0, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
                state[0, -1] = float(video_chunk_size) / float(delay) / M_IN_K # kilo byte / ms
                state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
                # last quality
                state[2, -1] = bitrate_versions[bit_rate] / float(np.max(bitrate_versions))  
                state[3, -1] = np.minimum(video_chunk_remain, total_chunk_num) / float(total_chunk_num)
                state[4 : 4 + br_dim, -1] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K # mega byte
                state[10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR

                ob[0, -1] = float(video_chunk_size) / float(delay) / M_IN_K # kilo byte / ms
                ob[1, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR # seconds
                # ob[1 : 1 + br_dim, -1] = np.array(curr_chunk_sizes) / M_IN_K / M_IN_K # mega byte
                # ob[1 + br_dim : 1 + 2 * br_dim, -1] = np.array(curr_chunk_psnrs) / DB_NORM_FACTOR

                # find the solution of robustMPC
                bit_rate_opt = expert.opt_action_robustMPC(state, args)

                # compute the nn-based choice for next video chunk
                ob_ = np.array([ob]).transpose(0, 2, 1)# [N, C_LEN, in_channels]
                ob_ = torch.from_numpy(ob_).type(dtype)

                state_ = np.array([state])
                state_ = torch.from_numpy(state_).type(dtype)
                with torch.no_grad():
                    latent = vae_net.get_latent(ob_).detach()
                    prob = model_actor(state_, latent)
                action = prob.multinomial(num_samples=1).detach()
                bit_rate_ = int(action.squeeze().cpu().numpy())
                
                # retrieve the starting status
                end_flag = end_of_video
                if end_of_video:
                    # define the observations for vae
                    # observations for vae input
                    ob = np.zeros((vae_in_channels, c_len)) 

                    # define the state for rl agent
                    state = np.zeros((s_info, s_len))

                    bit_rate_ = bit_rate_opt = last_bit_rate = DEFAULT_QUALITY
                    time_stamp, end_flag, video_chunk_remain = 0., True, total_chunk_num

                    expert.clear_buffer()
                    break

            ##===store the transitions and learn the model===
            # compute returns and GAE(lambda) advantages:
            if len(states) != len(rewards):
                if len(states) + 1 == len(rewards):
                    rewards = rewards[1:]
                else:
                    print('error in length of states!')
                    break
            # R = Variable(R)
            
            # obs_target.append(np.zeros((1 + 2 * br_dim, c_len)))
            memory.push([states, tar_actions, obs])
        
            ##==Network parameters update==
            model_actor.train()
            vae_net.train()
            if memory.return_size() >= 0.1 * memory.capacity:                
                # update the parameters
                vae_kld_loss_ = []
                policy_ce_loss_ = []
                policy_mi_loss_ = []
                for _ in range(UPDATE_PER_EPOCH):
                    # sample minibatch from the replay memory
                    batch_states, batch_tar_actions, batch_obs = \
                                                        memory.sample_cuda(minibatch_size)
                    # states_size = np.shape(batch_states)               
                    # action_size = np.shape(batch_tar_actions)
                    # obs_size = np.shape(batch_obs)
                    # assert states_size[1]==s_info and states_size[2]==s_len
                    # assert states_size[0] == action_size[0] and action_size[1] == 1
                    # assert obs_size[2] == c_len

                    ## learn the VAE network
                    # ------------------ VAE case -----------------------
                    batch_latents = vae_net.get_latent(batch_obs)

                    # latent samples for p(z_i|s)
                    # batch_s_ = batch_obs[:, -s_len:, :]

                    x_train = batch_obs # (N, C_LEN, in_channels)

                    # fit the model
                    z_mu, z_log_var = vae_net.forward(x_train)
                    kld_loss = vae_net.loss_function(z_mu, z_log_var)
                    vae_kld_loss_.append(kld_loss.detach().cpu().numpy())

                    # record loss infors


                    ## learn the RL policy network
                    sample_num = args.sp_n
                    latent_samples = []
                    for _ in range(sample_num):
                        latent_samples.append(
                                        torch.randn(minibatch_size, latent_dim).type(dtype)
                                        ) #.detach()


                    ## compute actor loss (cross entropy loss, entropy loss, and mutual information loss)
                    batch_actions = batch_tar_actions.unsqueeze(1)
                    probs_ = model_actor.forward(batch_states, batch_latents)
                    prob_value_ = torch.gather(probs_, dim=1, index=batch_actions)
                    cross_entropy = -torch.mean(torch.log(prob_value_ + 1e-6))
                    entropy = -torch.mean(probs_ * torch.log(probs_ + 1e-6))

                    # mutual information loss
                    probs_samples = torch.zeros(minibatch_size, br_dim, 1).type(dtype)
                    for idx in range(sample_num):
                        probs_ = model_actor(batch_states, latent_samples[idx])
                        probs_ = probs_.unsqueeze(2)
                        probs_samples = torch.cat((probs_samples, probs_), 2)
                    probs_samples = probs_samples[:, :, 1:]
                    probs_sa = torch.mean(probs_samples, dim=2) # p(a|s) = 1/L * \sum p(a|s, z_i) p(z_i|s)
                    probs_sa = Variable(probs_sa)
                    ent_noLatent = - torch.mean(probs_sa * torch.log(probs_sa + 1e-6))
                    mutual_info = ent_noLatent - entropy
                    loss_actor = - (coeff_alpha * -1 * cross_entropy + \
                                        coeff_beta * entropy + \
                                            coeff_gamma * mutual_info)
                    # loss_actor = - (cross_entropy + entropy + mutual_info)

                    # loss_actor = - (coeff_alpha * -1 * cross_entropy + coeff_beta * entropy)

                    optimizer_actor.zero_grad()
                    optimiser_vae.zero_grad()

                    ## compute the gradients
                    loss_total = loss_actor + kld_loss
                    loss_total.backward(retain_graph=False)

                    ## clip the gradients to aviod outputting inf or nan from the softmax function
                    clip_grad_norm_(model_actor.parameters(), \
                                        max_norm = MAX_GRAD_NORM, norm_type = 2)
                    clip_grad_norm_(vae_net.parameters(), \
                                        max_norm = MAX_GRAD_NORM, norm_type = 2)
                    ## update the parameters
                    optimizer_actor.step()
                    optimiser_vae.step()

                    # record the loss information
                    policy_ce_loss_.append(cross_entropy.detach().cpu().numpy())
                    policy_mi_loss_.append(mutual_info.detach().cpu().numpy())

                # ---- logging the loss information ----
                writer.add_scalar("Avg_VAE_kld_loss", np.mean(vae_kld_loss_), epoch)
                writer.add_scalar("Avg_loss_CE", np.mean(policy_ce_loss_), epoch)
                writer.add_scalar("Avg_loss_MI", np.mean(policy_mi_loss_), epoch)
                writer.flush()

                # update epoch counter
                epoch += 1

                # update the prior of latent
                # if epoch % args.prior_ui == 0 and args.prior_not:
                #     _, _, valid_obs, _, _ = memory.sample(minibatch_size)
                #     valid_obs = np.array(valid_obs[:-1]).transpose(0, 2, 1)# valid_obs[:-1]
                #     valid_obs = torch.from_numpy(valid_obs).type(dtype)

                #     z_mu_valid, z_log_var_valid = vae_net.encode(valid_obs)
                #     z_mu_valid = z_mu_valid.detach()
                #     z_log_var_valid = z_log_var_valid.detach()
                #     z_mu_posterior = torch.mean(z_mu_valid, dim= 0) # E[X] = \sum_i w_i \mu_i
                #     z_var_ = torch.mean(z_log_var_valid.exp(), dim = 0)
                #     z_mu_squared_ = torch.mean(z_mu_valid ** 2, dim = 0) 
                #     z_log_var_posterior = torch.log(z_var_ + z_mu_squared_ - z_mu_posterior ** 2) 
                #     # E[(X-\mu)^2] = \sum_i w_i (var_i + \mu_i^2) - \mu^2

                #     vae_net.update_prior(z_mu_posterior, z_log_var_posterior)
                    
                    

            ## test and save the model
            if epoch % args.valid_i == 0 and epoch > 0 :
                logging.info("Model saved in file")
                model_vae = vae_net
                valid(args, valid_env, model_actor, model_vae, epoch, \
                        test_log_file, add_str, summary_dir)
                # entropy_weight = 0.95 * entropy_weight
                # ent_coeff = 0.95 * ent_coeff

                # save models
                actor_save_path = summary_dir + '/' + add_str + \
                                    "/%s_%s_%d.model" %(str('policy'), add_str, int(epoch))
                vae_save_path = summary_dir + '/' + add_str + \
                                    "/%s_%s_%d.model" %(str('VAE'), add_str, int(epoch))
                if os.path.exists(actor_save_path): os.system('rm ' + actor_save_path)
                if os.path.exists(vae_save_path): os.system('rm ' + vae_save_path)
                torch.save(model_actor.state_dict(), actor_save_path)
                torch.save(vae_net.state_dict(), vae_save_path)

                # turn on the training model
                model_actor.train()
                vae_net.train()

    writer.close()
    model_vae = vae_net
    return model_actor.state_dict(), model_vae.state_dict()

# def main():
#     train_bmpc(1000)

# if __name__ == '__main__':
#     main()
