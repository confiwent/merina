import argparse
import sys
import numpy as np
sys.path.append('../envs/')
import fixed_env_log as env
import load_trace

A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
REBUF_PENALTY_lin = 4.3 #dB
REBUF_PENALTY_log = 2.66
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
MINIMUM_BUFFER_S = 10
BUFFER_TARGET_S = 30

LOG_FILE_OBE = '../Results/test/lin/oboe/log_test_bola'
LOG_FILE_3GP = '../Results/test/lin/3gp/log_test_bola'
LOG_FILE_FCC = '../Results/test/lin/fcc/log_test_bola'
LOG_FILE_FH = '../Results/test/lin/fh/log_test_bola'
LOG_FILE_PUF = '../Results/test/lin/puffer/log_test_bola'
LOG_FILE_PUF2 = '../Results/test/lin/puffer2/log_test_bola'

LOG_FILE_OBE_LOG = '../Results/test/log/oboe/log_test_bola'
LOG_FILE_3GP_LOG = '../Results/test/log/3gp/log_test_bola'
LOG_FILE_FCC_LOG = '../Results/test/log/fcc/log_test_bola'
LOG_FILE_FH_LOG = '../Results/test/log/fh/log_test_bola'
LOG_FILE_PUF_LOG = '../Results/test/log/puffer/log_test_bola'
LOG_FILE_PUF2_LOG = '../Results/test/log/puffer2/log_test_bola'

TEST_TRACES_FCC = '../envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = '../envs/traces/traces_oboe/'
TEST_TRACES_3GP = '../envs/traces/traces_3gp/'
TEST_TRACES_FH = '../envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_PUF = '../envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = '../envs/traces/puffer_220218/test_traces/'

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

parser = argparse.ArgumentParser(description='BOLA')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCC&3GP traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')

def main():
    args = parser.parse_args()
    
    video = 'Mao'
    # video = 'Avengers'
    video_size_file = '../envs/video_size/' + video + '/video_size_'
    video_psnr_file = '../envs/video_psnr/' + video + '/chunk_psnr'
    
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

    test_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), VIDEO_BIT_RATE, \
                            1, rebuff_p, SMOOTH_PENALTY, 0)

    log_path = log_file_init + '_' + all_file_names[test_env.trace_idx]
    log_file = open(log_path, 'w')

    _, _, _, total_chunk_num, \
            bitrate_versions, rebuffer_penalty, smooth_penalty = test_env.get_env_info()

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
    bit_rate = DEFAULT_QUALITY

    r_batch = []
    gp = 1 - 0 + (np.log(VIDEO_BIT_RATE[-1] / float(VIDEO_BIT_RATE[0])) - 0) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # log
    vp = MINIMUM_BUFFER_S/(0+ gp -1)
    # gp = 1 - VIDEO_BIT_RATE[0]/1000.0 + (VIDEO_BIT_RATE[-1]/1000. - VIDEO_BIT_RATE[0]/1000.) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # lin 
    # vp = MINIMUM_BUFFER_S/(VIDEO_BIT_RATE[0]/1000.0+ gp -1)
    

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain, \
                        _ = test_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
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

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # if buffer_size < RESEVOIR:
        #     bit_rate = 0
        # elif buffer_size >= RESEVOIR + CUSHION:
        #     bit_rate = A_DIM - 1
        # else:
        #     bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        score = -65535
        for q in range(len(VIDEO_BIT_RATE)):
            s = (vp * (np.log(VIDEO_BIT_RATE[q] / float(VIDEO_BIT_RATE[0])) + gp) - buffer_size) / next_video_chunk_sizes[q]
            # s = (vp * (VIDEO_BIT_RATE[q]/1000. + gp) - buffer_size) / next_video_chunk_sizes[q] # lin
            if s>=score:
                score = s
                bit_rate = q

        bit_rate = int(bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
            bit_rate = DEFAULT_QUALITY  # use the default action here

            time_stamp = 0

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = log_file_init + '_' + all_file_names[test_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
