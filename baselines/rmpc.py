"""
In this version, the RobustMPC is adopted to control the rate adaptation, with the harmonic mean bandwidth prediction method. quality metric: PSNR 
"""
import numpy as np
import sys
sys.path.append('../envs/')
import fixed_env_log as env
import load_trace
import matplotlib.pyplot as plt
import itertools
import time
import argparse


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 3
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
TOTAL_VIDEO_CHUNKS = 49
M_IN_K = 1000.0
REBUF_PENALTY_lin = 4.3 #dB
REBUF_PENALTY_log = 2.66
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000


LOG_FILE_OBE = '../Results/test/lin/oboe/log_test_mpc'
LOG_FILE_3GP = '../Results/test/lin/3gp/log_test_mpc'
LOG_FILE_FCC = '../Results/test/lin/fcc/log_test_mpc'
LOG_FILE_FH = '../Results/test/lin/fh/log_test_mpc'
LOG_FILE_PUF = '../Results/test/lin/puffer/log_test_mpc'
LOG_FILE_PUF2 = '../Results/test/lin/puffer2/log_test_mpc'

LOG_FILE_OBE_LOG = '../Results/test/log/oboe/log_test_mpc'
LOG_FILE_3GP_LOG = '../Results/test/log/3gp/log_test_mpc'
LOG_FILE_FCC_LOG = '../Results/test/log/fcc/log_test_mpc'
LOG_FILE_FH_LOG = '../Results/test/log/fh/log_test_mpc'
LOG_FILE_PUF_LOG = '../Results/test/log/puffer/log_test_mpc'
LOG_FILE_PUF2_LOG = '../Results/test/log/puffer2/log_test_mpc'

TEST_TRACES_FCC = '../envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = '../envs/traces/traces_oboe/'
TEST_TRACES_3GP = '../envs/traces/traces_3gp/'
TEST_TRACES_FH = '../envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_PUF = '../envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = '../envs/traces/puffer_220218/test_traces/'

parser = argparse.ArgumentParser(description='RobustMPC')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCC&3GP traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')

CHUNK_COMBO_OPTIONS = []
# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

class video_size(object):
    def __init__(self):
        self.video_sizes = {}

    def store_size(self, size_file_path):
        for bitrate in range(A_DIM):
            self.video_sizes[bitrate] = []
            with open(size_file_path + str(bitrate)) as f:
                for line in f:
                    self.video_sizes[bitrate].append(int(line.split()[0]))

    def get_chunk_size(self, quality, index):
        if ( index < 0 or index > TOTAL_VIDEO_CHUNKS-1 ):
            return 0
        # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
        # sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
        return self.video_sizes[quality][index]

def main():
    
    start = time.time()

    args = parser.parse_args()
    video = 'Mao'
    # video = 'Avengers'
    video_size_file = '../envs/video_size/' + video + '/video_size_'

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

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

    test_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), VIDEO_BIT_RATE, \
                            1, rebuff_p, SMOOTH_PENALTY, 0)

    log_path = log_file_init + '_' + all_file_names[test_env.trace_idx]
    log_file = open(log_path, 'w')

    _, _, _, total_chunk_num, \
            bitrate_versions, rebuffer_penalty, smooth_penalty = test_env.get_env_info()

    chunk_size_info = video_size()
    chunk_size_info.store_size(video_size_file)

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    harmonic_bandwidth = 0 
    future_bandwidth = 0

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    # entropy_record = []

    # make chunk combination options
    for combo in itertools.product([0,1,2,3,4,5], repeat=MPC_FUTURE_CHUNK_COUNT):
        CHUNK_COMBO_OPTIONS.append(combo)

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

        r_batch.append(reward)

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
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / \
                                    float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)
        # past_bandwidth_ests.append(future_bandwidth)


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain -1)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNKS - last_index < MPC_FUTURE_CHUNK_COUNT):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        #start = time.time()
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_bit_rate_ = last_bit_rate
            for position in range(0, len(combo)): 
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                chunk_action = combo[position]
                chunk_quality_ = np.log(bitrate_versions[chunk_action] / \
                                    float(bitrate_versions[0])) if args.log else \
                                        bitrate_versions[chunk_action] / M_IN_K
                download_time = (chunk_size_info.get_chunk_size(chunk_action, index)/1000000.)/\
                                    future_bandwidth # this is MB/MB/s --> seconds
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += chunk_quality_
                # smoothness_diffs_p += max(chunk_quality_ - last_quality_, 0)
                # smoothness_diffs_n += max(last_quality_ - chunk_quality_, 0)
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                last_quality_ = np.log(bitrate_versions[last_bit_rate_] / \
                                    float(bitrate_versions[0])) if args.log else \
                                        bitrate_versions[last_bit_rate_] / M_IN_K
                smoothness_diffs += abs(chunk_quality_ - last_quality_)
                last_bit_rate_ = chunk_action
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
            
            reward = bitrate_sum - (rebuff_p*curr_rebuffer_time) - \
                        SMOOTH_PENALTY * smoothness_diffs

            if ( reward >= max_reward ):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
                if ( best_combo != () ): # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del past_bandwidth_ests[:]
            del past_errors[:]

            time_stamp = 0

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                end = time.time()
                print(end - start)
                break

            log_path = log_file_init + '_' + all_file_names[test_env.trace_idx]
            log_file = open(log_path, 'w')

            end = time.time()
            print(end - start)


if __name__ == '__main__':
    main()

