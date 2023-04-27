import os
import argparse
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
import libcomyco_lin as libcomyco

sys.path.append('../envs/')
import fixed_env_log as env
import load_trace
from functools import reduce
from operator import mul

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
M_IN_K = 1000.0
REBUF_PENALTY_lin = 4.3 #dB
REBUF_PENALTY_log = 2.66
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL_LIN = '../models/comyco/lin/nn_model_ep_580.ckpt' # for lin 
NN_MODEL_LOG = '../models/comyco/log/nn_model_ep_680.ckpt' # for log

LOG_FILE_OBE = '../Results/test/lin/oboe/log_test_cmc'
LOG_FILE_3GP = '../Results/test/lin/3gp/log_test_cmc'
LOG_FILE_FCC = '../Results/test/lin/fcc/log_test_cmc'
LOG_FILE_FH = '../Results/test/lin/fh/log_test_cmc'
LOG_FILE_PUF = '../Results/test/lin/puffer/log_test_cmc'
LOG_FILE_PUF2 = '../Results/test/lin/puffer2/log_test_cmc'

LOG_FILE_OBE_LOG = '../Results/test/log/oboe/log_test_cmc'
LOG_FILE_3GP_LOG = '../Results/test/log/3gp/log_test_cmc'
LOG_FILE_FCC_LOG = '../Results/test/log/fcc/log_test_cmc'
LOG_FILE_FH_LOG = '../Results/test/log/fh/log_test_cmc'
LOG_FILE_PUF_LOG = '../Results/test/log/puffer/log_test_cmc'
LOG_FILE_PUF2_LOG = '../Results/test/log/puffer2/log_test_cmc'

TEST_TRACES_FCC = '../envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = '../envs/traces/traces_oboe/'
TEST_TRACES_3GP = '../envs/traces/traces_3gp/'
TEST_TRACES_FH = '../envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_PUF = '../envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = '../envs/traces/puffer_220218/test_traces/'

parser = argparse.ArgumentParser(description='Comyco-MM21')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCC&3GP traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    args = parser.parse_args()
    video = 'Mao'
    # video = 'Avengers'
    video_size_file = '../envs/video_size/' + video + '/video_size_'

    # -----------------------------initialize the environment----------------------------------------
    if args.tf:
        test_traces = TEST_TRACES_FCC
        log_file_init = LOG_FILE_FCC_LOG if args.log else LOG_FILE_FCC
    elif args.tfh:
        test_traces = TEST_TRACES_FH
        log_file_init = LOG_FILE_FH_LOG if args.log else LOG_FILE_FH
    elif args.to:
        test_traces = TEST_TRACES_OBE
        log_file_init = LOG_FILE_OBE_LOG if args.log else LOG_FILE_OBE
    elif args.t3g:
        test_traces = TEST_TRACES_3GP
        log_file_init = LOG_FILE_3GP_LOG if args.log else LOG_FILE_3GP
    elif args.tp:
        test_traces = TEST_TRACES_PUF
        log_file_init = LOG_FILE_PUF_LOG if args.log else LOG_FILE_PUF
    elif args.tp2:
        test_traces = TEST_TRACES_PUF2
        log_file_init = LOG_FILE_PUF2_LOG if args.log else LOG_FILE_PUF2
    
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
    test_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, all_file_names = all_file_names,
                                video_size_file = video_size_file)

    # determine the QoE metric \
    rebuff_p = REBUF_PENALTY_log if args.log else REBUF_PENALTY_lin

    test_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), \
                            VIDEO_BIT_RATE, 1, rebuff_p, SMOOTH_PENALTY, 0)

    log_path = log_file_init + '_' + all_file_names[test_env.trace_idx]
    log_file = open(log_path, 'w')

    _, _, _, total_chunk_num, \
            bitrate_versions, rebuffer_penalty, smooth_penalty = test_env.get_env_info()

    with tf.Session() as sess:
        actor = libcomyco.libcomyco(sess,
                S_INFO, S_LEN, A_DIM, LR_RATE = 1e-4)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL_LIN is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL_LOG) if args.log else \
                    saver.restore(sess, NN_MODEL_LIN)
            print("Testing model restored.")

        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print "num", num_params

        time_stamp = 0

        bit_rate = DEFAULT_QUALITY
        last_bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain, \
                        _ = test_env.get_video_chunk(int(bit_rate))


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
            r_batch.append(reward)
            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob, _ = actor.predict(
                np.reshape(state, (-1, S_INFO, S_LEN)))
            bit_rate = np.argmax(action_prob[0])

            s_batch.append(state)

            entropy_record.append(actor.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_bit_rate = DEFAULT_QUALITY

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = log_file_init + '_' + all_file_names[test_env.trace_idx]
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
