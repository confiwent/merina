import time
from collections import deque
import numpy as np
# from numba import jit
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from .beta_vae_v6 import BetaVAE
from .AC_net_v6 import Actor

RANDOM_SEED = 42
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
DB_NORM_FACTOR = 25.0
BITS_IN_BYTE = 8.0
M_IN_K = 1000.0
DEFAULT_QUALITY = 1

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor

# @jit(forceobj=True)
def evaluation(actor_net, vae_net, log_path_ini, test_env, \
    s_info, s_len, c_len, total_chunk_num, \
        bitrate_versions, rebuffer_penalty, smooth_penalty, a_dim, args):
    
    # define the observations for vae
    vae_in_channels = 2
    ob = np.zeros((vae_in_channels, c_len)) # observations for vae input, 

    # define the state for rl agent
    state = np.zeros((s_info, s_len))
    # state[-1][-1] = 1.0
    # state = torch.from_numpy(state)
    bit_rate = DEFAULT_QUALITY
    last_bit_rate = DEFAULT_QUALITY
    time_stamp = 0.

    # model.load_state_dict(model.state_dict())
    all_file_name = test_env.all_file_names
    log_path = log_path_ini + '_' + all_file_name[test_env.trace_idx]
    log_file = open(log_path, 'w')
    time_stamp = 0
    for video_count in tqdm(range(len(all_file_name))):
        while True:
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain, \
                        _ = test_env.get_video_chunk(bit_rate)
            
            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smooth penalty
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
            last_bit_rate = bit_rate

            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(bitrate_versions[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(reward) + '\n')
            log_file.flush()

            # dequeue history record
            state = np.roll(state, -1, axis=1)
            ob = np.roll(ob, -1, axis=1)

            # this should be S_INFO number of terms
            # state[0, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
            state[0, -1] = float(video_chunk_size) / float(delay) / M_IN_K # kilo byte / ms
            state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
            state[2, -1] = bitrate_versions[bit_rate] / float(np.max(bitrate_versions))  # last quality
            state[3, -1] = np.minimum(video_chunk_remain, total_chunk_num) / float(total_chunk_num)
            state[4 : 4 + a_dim, -1] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K# mega byte
            state[10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR

            ob[0, -1] = float(video_chunk_size) / float(delay) / M_IN_K # kilo byte / ms
            ob[1, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR# seconds
            # ob[2 : 2 + a_dim, -1] = np.array(curr_chunk_sizes) / M_IN_K / M_IN_K# mega byte
            # ob[1 + a_dim : 1 + 2 * a_dim, -1] = np.array(curr_chunk_psnrs) / DB_NORM_FACTOR

            ob_ = np.array([ob]).transpose(0, 2, 1)
            ob_ = torch.from_numpy(ob_).type(dtype)

            state_ = np.array([state])
            state_ = torch.from_numpy(state_).type(dtype)

            with torch.no_grad():
                latent = vae_net.get_latent(ob_).detach()
                prob = actor_net.forward(state_, latent).detach()
            
            ## NOTICES: you can choose either the greedy or stochastic policy output as you like
            
            if args.stocha:
                # stochastic policy
                action = prob.multinomial(num_samples=1).detach()
                bit_rate = int(action.squeeze().cpu().numpy())
            else:
                # greedy policy
                bit_rate = int(torch.argmax(prob).squeeze().cpu().numpy())
            
            if end_of_video:
                ob = np.zeros((vae_in_channels, c_len)) 
                # observations for vae input

                # define the state for rl agent
                state = np.zeros((s_info, s_len))
                # state[-1][-1] = 1.0
                bit_rate = last_bit_rate = DEFAULT_QUALITY
                log_file.write('\n')
                log_file.close()
                time_stamp = 0

                if video_count + 1 >= len(all_file_name):
                    break
                else:
                    log_path = log_path_ini + '_' + all_file_name[test_env.trace_idx]
                    log_file = open(log_path, 'w')
                    break

def valid(args, env, actor_net, vae_net, epoch, log_file, add_str, summary_dir):
    summary_dir_ = summary_dir + '/' + add_str + '/test_results'
    os.system('rm -r ' + summary_dir_)
    os.system('mkdir ' + summary_dir_)

    log_path_ini = summary_dir_ + '/log_valid_' + add_str

    # Get envs informations
    s_info, s_len, c_len, total_chunk_num, \
        bitrate_versions, rebuffer_penalty, smooth_penalty = env.get_env_info()
    a_dim = len(bitrate_versions)

    actor_net.eval()
    vae_net.eval()

    evaluation(actor_net, vae_net, log_path_ini, env, \
                    s_info, s_len, c_len, total_chunk_num, \
                        bitrate_versions, rebuffer_penalty, smooth_penalty, a_dim, args)

    rewards = []
    test_log_folder = summary_dir_ + '/'
    test_log_files = os.listdir(test_log_folder)
    for test_log_file in test_log_files:
        reward = []
        with open(test_log_folder + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[4:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(int(epoch)) + '\t' +
                str(rewards_min) + '\t' +
                str(rewards_5per) + '\t' +
                str(rewards_mean) + '\t' +
                str(rewards_median) + '\t' +
                str(rewards_95per) + '\t' +
                str(rewards_max) + '\n')
    log_file.flush()

    # actor_save_path = summary_dir + '/' + add_str + \
    #   "/%s_%s_%d.model" %(str('policy'), add_str, int(epoch))
    # vae_save_path = summary_dir + '/' + add_str + \
    #   "/%s_%s_%d.model" %(str('VAE'), add_str, int(epoch))
    # if os.path.exists(actor_save_path): os.system('rm ' + actor_save_path)
    # if os.path.exists(vae_save_path): os.system('rm ' + vae_save_path)
    # torch.save(actor_net.state_dict(), actor_save_path)
    # torch.save(vae_net.state_dict(), vae_save_path)

def test(args, test_model, env, log_file, add_str, summary_dir):
    summary_dir_ = summary_dir
    # Get envs informations
    s_info, s_len, c_len, total_chunk_num, \
            bitrate_versions, rebuffer_penalty, smooth_penalty = env.get_env_info()
    a_dim = len(bitrate_versions)

    latent_dim = args.latent_dim

    model_actor = Actor(a_dim, latent_dim, s_info, s_len).type(dtype)
    model_vae = BetaVAE(in_channels=2, hist_dim=c_len, latent_dim=latent_dim).type(dtype)
    model_actor.eval()
    model_vae.eval()
    model_actor.load_state_dict(torch.load(test_model[0]))
    model_vae.load_state_dict(torch.load(test_model[1]))

    # total_num_actor = sum(p.numel() for p in model_actor.parameters() if p.requires_grad)
    # print(total_num_actor)

    # total_num_vae = sum(p.numel() for p in model_vae.parameters() if p.requires_grad)
    # print(total_num_vae)

    # for name, paras in model_actor.named_parameters():
    #     print(name, ":", paras.size())

    # for name, paras in model_vae.named_parameters():
    #     print(name, ":", paras.size())

    evaluation(model_actor, model_vae, log_file, env, \
        s_info, s_len, c_len, total_chunk_num, \
                bitrate_versions, rebuffer_penalty, smooth_penalty, a_dim, args)
    
    rewards = []
    test_log_folder = summary_dir_
    test_log_files = os.listdir(test_log_folder)
    for test_log_file in test_log_files:
        if add_str in test_log_file:
            reward = []
            with open(test_log_folder + test_log_file, 'r') as f:
                for line in f:
                    parse = line.split()
                    try:
                        reward.append(float(parse[-1]))
                    except IndexError:
                        break
            rewards.append(np.mean(reward[4:]))

    rewards = np.array(rewards)
    print("mean_QoE: ", np.mean(rewards))
