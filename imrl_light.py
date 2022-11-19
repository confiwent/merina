import os
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim

from config.merina import args_merina
# from algos.train_ppo_v5 import train_ppo_v5
# from algos.train_dppo import train_dppo_pure
from algos.train_im_v4_light import train_iml_v4
from algos.test_v5_light import test
from algos.train_ppo_v6_light import train_ppo_v6
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

TEST_LOG_FILE_FCC = './Results/test/fcc/'
TEST_LOG_FILE_OBE = './Results/test/oboe/'
TEST_LOG_FILE_3GP = './Results/test/3gp/'
TEST_LOG_FILE_FH = './Results/test/fh/'
TEST_LOG_FILE_GHT = './Results/test/ghent/'
TEST_LOG_FILE_FHN = './Results/test/fh_noisy/'
TEST_LOG_FILE_PUF = './Results/test/puffer/'
TEST_LOG_FILE_PUF2 = './Results/test/puffer2/'

TEST_LOG_FILE_OBE_LOG = './Results/test/log/oboe/'
TEST_LOG_FILE_3GP_LOG = './Results/test/log/3gp/'
TEST_LOG_FILE_FCC_LOG = './Results/test/log/fcc/'
TEST_LOG_FILE_FH_LOG = './Results/test/log/fh/'
TEST_LOG_FILE_GHT_LOG = './Results/test/log/ghent/' # 4g
TEST_LOG_FILE_FHN_LOG = './Results/test/log/fh_noisy/'
TEST_LOG_FILE_PUF_LOG = './Results/test/log/puffer/'
TEST_LOG_FILE_PUF2_LOG = './Results/test/log/puffer2/'

# TEST_TRACES_FCC = './envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_FCC = './envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = './envs/traces/traces_oboe/'
TEST_TRACES_3GP = './envs/traces/traces_3gp/'
TEST_TRACES_GHT = './envs/traces/test_traces_4g2/'
TEST_TRACES_FHN = './envs/traces/test_traces_noisy/'
TEST_TRACES_FH = './envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_PUF = './envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = './envs/traces/puffer_220218/test_traces/'

# use FCC and HSDPA datasets to jointly train the models  
TRAIN_TRACES = './envs/traces/pre_webget_1608/cooked_traces/'
VALID_TRACES = './envs/traces/pre_webget_1608/test_traces/'

ADP_TRAIN_TRACES = './envs/traces/puffer_211017/cooked_traces/'
ADP_VALID_TRACES = './envs/traces/puffer_211017/test_traces/'

SUMMARY_DIR = './Results/sim'
MODEL_DIR = './saved_models'

#lin 
TEST_MODEL_ACT_MRL = './Results/sim/merina_lin2/policy_merina_lin2_200.model'
TEST_MODEL_VAE_MRL = './Results/sim/merina_lin2/VAE_merina_lin2_200.model'

#log
TEST_MODEL_ACT_MRL_LOG = './saved_models/0404/log/policy_imrl_log_440.model'
TEST_MODEL_VAE_MRL_LOG = './saved_models/0404/log/VAE_imrl_log_440.model'

def main():
    parser = argparse.ArgumentParser()
    _, rest_args = parser.parse_known_args() 
    args = args_merina.get_args(rest_args)

    rebuff_p = REBUF_PENALTY_log if args.log else REBUF_PENALTY_lin
    video_size_file = './envs/video_size/Mao/video_size_' #video = 'Avengers'


    if args.test:
        run_test(args, rebuff_p, video_size_file)
    else:
        run_train(args, rebuff_p, video_size_file)

def get_test_traces(args):
    if args.tf:
        log_save_dir = TEST_LOG_FILE_FCC_LOG if args.log else TEST_LOG_FILE_FCC
        test_traces = TEST_TRACES_FCC
    elif args.t3g:
        log_save_dir = TEST_LOG_FILE_3GP_LOG if args.log else TEST_LOG_FILE_3GP
        test_traces = TEST_TRACES_3GP
    elif args.to:
        log_save_dir = TEST_LOG_FILE_OBE_LOG if args.log else TEST_LOG_FILE_OBE
        test_traces = TEST_TRACES_OBE
    elif args.tg:
        log_save_dir = TEST_LOG_FILE_GHT_LOG if args.log else TEST_LOG_FILE_GHT
        test_traces = TEST_TRACES_GHT
    elif args.tn:
        log_save_dir = TEST_LOG_FILE_FHN_LOG if args.log else TEST_LOG_FILE_FHN
        test_traces = TEST_TRACES_FHN
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
    
    log_path = log_save_dir + 'log_test_' + args.name

    return log_save_dir, test_traces, log_path

def run_test(args, rebuff_p, video_size_file):
    log_save_dir, test_traces, log_path = get_test_traces(args)
    
    if not os.path.exists(log_save_dir):
            os.mkdir(log_save_dir)

    # log_save_dir = TEST_LOG_FILE_FCC
    # log_path = TEST_LOG_FILE_FCC + 'log_test_' + add_str
    # test_traces = TEST_TRACES_FCC
    # if not os.path.exists(log_save_dir):
    #     os.mkdir(log_save_dir)

    # determine the QoE metric \
    
    
    test_model_ = [TEST_MODEL_ACT_MRL_LOG, TEST_MODEL_VAE_MRL_LOG] if args.log \
                        else [TEST_MODEL_ACT_MRL, TEST_MODEL_VAE_MRL] 
    
    # video_size_file = './envs/video_size/Avengers/video_size_'

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
    test_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw, \
                                            all_file_names = all_file_names, \
                                                video_size_file = video_size_file)
    test_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, \
                            1, rebuff_p, SMOOTH_PENALTY, 0)
    
    test(args, test_model_, test_env, log_path)

def pre_training(args, train_env, valid_env, log_dir_path):
    add_str = args.name
    model_actor_para, model_vae_para = train_iml_v4(
                                                IMITATION_TRAIN_EPOCH, 
                                                train_env, 
                                                valid_env, 
                                                args, 
                                                add_str, 
                                                log_dir_path
    )

    # ##===== save models in the First stage
    model_save_dir = MODEL_DIR + '/' + add_str
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    # command = 'rm ' + SUMMARY_DIR + add_str + '/*'5
    # os.system(command)
    model_actor_save_path = model_save_dir + "/%s_%s_%d.model" %(\
                            str('Policy'), add_str, int(IMITATION_TRAIN_EPOCH))
    model_vae_save_path = model_save_dir + "/%s_%s_%d.model" %(\
                            str('VAE'), add_str, int(IMITATION_TRAIN_EPOCH))
    if os.path.exists(model_actor_save_path): os.system('rm ' + model_actor_save_path)
    if os.path.exists(model_vae_save_path): os.system('rm ' + model_vae_save_path)
    torch.save(model_actor_para, model_actor_save_path)
    torch.save(model_vae_para, model_vae_save_path)

    ## COPY THE LOG FILE
    os.system('cp ' + log_dir_path + '/' + add_str + '/log_test ' + model_save_dir + '/')

    return model_actor_save_path, model_vae_save_path

def run_train(args, rebuff_p, video_size_file):
    add_str = args.name
    log_dir_path = SUMMARY_DIR

    ##=== environments configures============
    if args.adp:
        Train_traces = ADP_TRAIN_TRACES
        Valid_traces = ADP_VALID_TRACES
    else:
        Train_traces = TRAIN_TRACES
        Valid_traces = VALID_TRACES
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Valid_traces)
    valid_env = env_test.Environment(
                                all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, 
                                all_file_names = all_file_names, 
                                video_size_file = video_size_file
    )
    valid_env.set_env_info(S_INFO, S_LEN, C_LEN, 
                        TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 
                        1, rebuff_p, SMOOTH_PENALTY, 0)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
    im_train_env = env.Environment(
                            all_cooked_time=all_cooked_time, 
                            all_cooked_bw=all_cooked_bw, 
                            video_size_file= video_size_file
    )
    im_train_env.set_env_info(S_INFO, S_LEN, C_LEN, 
                    TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 
                    1, rebuff_p, SMOOTH_PENALTY, 0)

    
    train_env = env.Environment(
                            all_cooked_time=all_cooked_time, 
                            all_cooked_bw=all_cooked_bw, 
                            video_size_file= video_size_file
    )
    train_env.set_env_info(S_INFO, S_LEN, C_LEN, 
                    TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 
                    1, rebuff_p, SMOOTH_PENALTY, 0)
    
   

    ## ==== Train MERINA =====
    if args.non_il:
        model_vae_save_path = './saved_models/im7/VAE_im7_1050.model'
        model_actor_save_path = './saved_models/im7/Policy_im7_1050.model'
        # model_vae_save_path = './saved_models/imrl_log/VAE_imrl_log_1050.model'
        # model_actor_save_path = './saved_models/imrl_log/Policy_imrl_log_1050.model'

        command = 'rm ' + log_dir_path + '/' + add_str + '/*'
        os.system(command)
        command = 'rm ' + log_dir_path + '/' + add_str + '/models/*'
        os.system(command)
    else:    
        model_actor_save_path, model_vae_save_path = pre_training(
                                                            args, im_train_env, 
                                                            valid_env, 
                                                            log_dir_path
        )

    if args.adp:
        # model_actor_save_path = './saved_models/0404/log/policy_imrl_log_440.model'
        # model_vae_save_path = './saved_models/0404/log/VAE_imrl_log_440.model'
        model_actor_save_path = './saved_models/0325/policy_imrl_1680.model'
        model_vae_save_path = './saved_models/0325/VAE_imrl_1680.model'

    # RL part
    model_vae_para = torch.load(model_vae_save_path)
    model_actor_para = torch.load(model_actor_save_path)

    train_ppo_v6(model_actor_para, model_vae_para, 
            train_env, valid_env, args, add_str, log_dir_path)

if __name__ == '__main__':
    main()