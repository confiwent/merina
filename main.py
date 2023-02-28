import os
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim

from algos.train_im_v4 import train_iml_v4
from algos.test_v5 import test
from algos.train_ppo_v6 import train_ppo_v6
import envs.env_log as env
import envs.fixed_env_log as env_test
from envs import load_trace

# Parameters of envs
S_INFO = 11 # 
S_LEN = 2 # maximum length of states 
C_LEN = 8 # content length 
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # kbps
TOTAL_CHUNK_NUM = 49
REBUF_PENALTY_lin = 4.3 #dB
REBUF_PENALTY_log = 2.66
SMOOTH_PENALTY = 1

IMITATION_TRAIN_EPOCH = 1050

## 3GP and OBE is the dataset of HSDPA and Oboe, respectively.

TEST_LOG_FILE_FCC = './Results/test/lin/fcc/'
TEST_LOG_FILE_OBE = './Results/test/lin/oboe/'
TEST_LOG_FILE_3GP = './Results/test/lin/3gp/'
TEST_LOG_FILE_FH = './Results/test/lin/fh/'
TEST_LOG_FILE_PUF = './Results/test/lin/puffer/'
TEST_LOG_FILE_PUF2 = './Results/test/lin/puffer2/'

TEST_LOG_FILE_OBE_LOG = './Results/test/log/oboe/'
TEST_LOG_FILE_3GP_LOG = './Results/test/log/3gp/'
TEST_LOG_FILE_FCC_LOG = './Results/test/log/fcc/'
TEST_LOG_FILE_FH_LOG = './Results/test/log/fh/'
TEST_LOG_FILE_PUF_LOG = './Results/test/log/puffer/'
TEST_LOG_FILE_PUF2_LOG = './Results/test/log/puffer2/'

TEST_TRACES_FCC = './envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = './envs/traces/traces_oboe/'
TEST_TRACES_3GP = './envs/traces/traces_3gp/'
TEST_TRACES_FH = './envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_PUF = './envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = './envs/traces/puffer_220218/test_traces/'


# use FCC and HSDPA datasets to jointly train the models  
TRAIN_TRACES = './envs/traces/fcc_and_hsdpa/cooked_traces/'
VALID_TRACES = './envs/traces/fcc_and_hsdpa/test_traces/'

ADP_TRAIN_TRACES = './envs/traces/puffer_211017/cooked_traces/'
ADP_VALID_TRACES = './envs/traces/puffer_211017/test_traces/'

SUMMARY_DIR = './Results/sim'
MODEL_DIR = './models'

#lin 
TEST_MODEL_ACT_LIN = './models/pretrain_policy_lin.model'
TEST_MODEL_VAE_LIN = './models/pretrain_vae_lin.model'

#log
TEST_MODEL_ACT_LOG = './models/pretrain_policy_log.model'
TEST_MODEL_VAE_LOG = './models/pretrain_vae_log.model'

parser = argparse.ArgumentParser(description='MRL-based ABR')
parser.add_argument('--test', action='store_true', help='Evaluate only')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--adp', action='store_true', help='Meta adaptation')
parser.add_argument('--stocha', action='store_true', help='greedy or stochastic policy')
parser.add_argument('--name', default='merina', help='the name of result folder')
parser.add_argument('--latent-dim', nargs='?', const=64, default=64, type=int, help='The dimension of latent space')
parser.add_argument('--mpc-h', nargs='?', const=5, default=5, type=int, help='The MPC planning horizon')
parser.add_argument('--valid-i',nargs='?', const=100, default=100, type=int, help='The valid interval during the training')
parser.add_argument('--kld-beta', nargs='?', const=0.01, default=0.01, type=float, help='The coefficient of kld in the VAE loss function')
parser.add_argument('--kld-lambda', nargs='?', const=1.1, default=1.1, type=float, help='The coefficient of kld in the VAE recon loss function') ## control the strength of over-fitting of reconstruction, KL divergence between the prior P(D) and the distribution of P(D|\theta)
parser.add_argument('--vae-gamma', nargs='?', const=0.7, default=0.7, type=float, help='The coefficient of reconstruction loss in the VAE loss function')
parser.add_argument('--lc-alpha', nargs='?', const=1, default=1, type=float, help='The coefficient of cross entropy in the actor loss function')
parser.add_argument('--lc-beta', nargs='?', const=0.2, default=0.2, type=float, help='The coefficient of entropy in the imitation loss function')
parser.add_argument('--lc-mu', nargs='?', const=0.1, default=0.1, type=float, help='The coefficient of cross entropy in the actor loss function')
parser.add_argument('--lc-gamma', nargs='?', const=0.1, default=0.1, type=float, help='The coefficient of mutual information in the actor loss function')
parser.add_argument('--sp-n', nargs='?', const=10, default=10, type=int, help='The sample numbers of the mutual information')
parser.add_argument('--gae-gamma', nargs='?', const=0.99, default=0.99, type=float, help='The gamma coefficent for GAE estimation')
parser.add_argument('--gae-lambda', nargs='?', const=0.95, default=0.95, type=float, help='The lambda coefficent for GAE estimation')
parser.add_argument('--batch-size', nargs='?', const=64, default=64, type=int, help='Minibatch size for training')
parser.add_argument('--ppo-ups', nargs='?', const=2, default=2, type=int, help='Update numbers in each epoch for PPO')
parser.add_argument('--explo-num', nargs='?', const=32, default=32, type=int, help='Exploration steps for roll-out')
parser.add_argument('--ro-len', nargs='?', const=25, default=25, type=int, help='Length of roll-out')
parser.add_argument('--clip', nargs='?', const=0.04, default=0.04, type=float, help='Clip value of ppo')
parser.add_argument('--anneal-p', nargs='?', const=0.95, default=0.95, type=float, help='Annealing parameters for entropy regularization')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCC_and_HSDPA traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--t3g', action='store_true', help='Use HSDPA traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer-211017 traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer-220218 traces')


def main():
    args = parser.parse_args()
    add_str = args.name
    if args.tf:
        log_save_dir = TEST_LOG_FILE_FCC_LOG if args.log else TEST_LOG_FILE_FCC
        test_traces = TEST_TRACES_FCC
    elif args.t3g:
        log_save_dir = TEST_LOG_FILE_3GP_LOG if args.log else TEST_LOG_FILE_3GP
        test_traces = TEST_TRACES_3GP
    elif args.to:
        log_save_dir = TEST_LOG_FILE_OBE_LOG if args.log else TEST_LOG_FILE_OBE
        test_traces = TEST_TRACES_OBE
    elif args.tp:
        log_save_dir = TEST_LOG_FILE_PUF_LOG if args.log else TEST_LOG_FILE_PUF
        test_traces = TEST_TRACES_PUF
    elif args.tp2:
        log_save_dir = TEST_LOG_FILE_PUF2_LOG if args.log else TEST_LOG_FILE_PUF2
        test_traces = TEST_TRACES_PUF2
    elif args.tfh:
        log_save_dir = TEST_LOG_FILE_FH_LOG if args.log else TEST_LOG_FILE_FH
        test_traces = TEST_TRACES_FH
    else:
        # print("Please choose the throughput data traces!!!")
        log_save_dir = TEST_LOG_FILE_FCC_LOG if args.log else TEST_LOG_FILE_FCC
        test_traces = TEST_TRACES_FCC

    log_path = log_save_dir + 'log_test_' + add_str

    if not os.path.exists(log_save_dir):
            os.mkdir(log_save_dir)

    # log_save_dir = TEST_LOG_FILE_FCC
    # log_path = TEST_LOG_FILE_FCC + 'log_test_' + add_str
    # test_traces = TEST_TRACES_FCC
    # if not os.path.exists(log_save_dir):
    #     os.mkdir(log_save_dir)

    # determine the QoE metric \
    rebuff_p = REBUF_PENALTY_log if args.log else REBUF_PENALTY_lin

    test_model_ = [TEST_MODEL_ACT_LOG, TEST_MODEL_VAE_LOG] if args.log \
                        else [TEST_MODEL_ACT_LIN, TEST_MODEL_VAE_LIN] 
    video_size_file = './envs/video_size/Mao/video_size_' #video size file
    # video_size_file = './envs/video_size/Avengers/video_size_'

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
    test_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw, \
                                            all_file_names = all_file_names, \
                                                video_size_file = video_size_file)
    test_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, \
                            1, rebuff_p, SMOOTH_PENALTY, 0)

    if args.test:
        test(args, test_model_, test_env, log_path, add_str, log_save_dir)
    else:
        log_dir_path = SUMMARY_DIR
        if args.adp:
            Train_traces = ADP_TRAIN_TRACES
            Valid_traces = ADP_VALID_TRACES
        else:
            Train_traces = TRAIN_TRACES
            Valid_traces = VALID_TRACES
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Valid_traces)
        valid_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                    all_cooked_bw=all_cooked_bw, all_file_names = all_file_names, video_size_file = video_size_file)
        valid_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 1, rebuff_p, SMOOTH_PENALTY, 0)

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file)
        train_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 1, rebuff_p, SMOOTH_PENALTY, 0)

        model_actor_para, model_vae_para = train_iml_v4(IMITATION_TRAIN_EPOCH, train_env, valid_env, args, add_str, log_dir_path)

        # ##===== save models in the First stage
        model_save_dir = MODEL_DIR + '/' + add_str
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        # command = 'rm ' + SUMMARY_DIR + add_str + '/*'5
        # os.system(command)
        model_actor_save_path = model_save_dir + "/%s_%s_%d.model" %(str('Policy'), add_str, int(IMITATION_TRAIN_EPOCH))
        model_vae_save_path = model_save_dir + "/%s_%s_%d.model" %(str('VAE'), add_str, int(IMITATION_TRAIN_EPOCH))
        if os.path.exists(model_actor_save_path): os.system('rm ' + model_actor_save_path)
        if os.path.exists(model_vae_save_path): os.system('rm ' + model_vae_save_path)
        torch.save(model_actor_para, model_actor_save_path)
        torch.save(model_vae_para, model_vae_save_path)

        if args.adp:
            model_actor_save_path = './models/pretrain_policy_log.model'
            model_vae_save_path = './models/pretrain_vae_log.model'

        # RL part
        model_vae_para = torch.load(model_vae_save_path)
        model_actor_para = torch.load(model_actor_save_path)

        train_ppo_v6(model_actor_para, model_vae_para, train_env, valid_env, args, add_str, log_dir_path)


if __name__ == '__main__':
    main()