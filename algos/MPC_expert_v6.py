"""
In this version, the MPC is adopted to control the rate adaptation, with the future bandwidth having been known in advance. So we call this version MPC-Oracal
"""
import numpy as np
import random
import itertools
# from .mpc_pruning import solving_opt

MPC_FUTURE_CHUNK_COUNT = 7
M_IN_K = 1000.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000


class ABRExpert:
    ''' a MPC-based planning method to optimize the expected returns in adaptive video streaming, with the throughput dynamics being known in advance '''
    def __init__(self, abr_env, rebuffer_penalty, smooth_penalty, \
                    mpc_horizon = MPC_FUTURE_CHUNK_COUNT, total_chunk_num = 48, \
                        bitrate_version = [300,750,1200,1850,2850,4300]):
        self.env = abr_env
        self.rebuffer_penalty = rebuffer_penalty
        self.smooth_penalty = smooth_penalty
        self.mpc_horizon = mpc_horizon
        self.total_chunk_num = total_chunk_num
        self.video_chunk_remain = total_chunk_num
        self.time_stamp = 0
        self.start_buffer = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.bit_rate = DEFAULT_QUALITY
        self.bitrate_versions = bitrate_version

        self.CHUNK_COMBO_OPTIONS = []
        self.past_errors = []  # past errors in bandwidth
        self.past_bandwidth_ests = []

        # make chunk combination options
        for combo in itertools.product([0,1,2,3,4,5], repeat=mpc_horizon):
            self.CHUNK_COMBO_OPTIONS.append(combo)

    def optimal_action(self, args):
        # future chunks length (try 4 if that many remaining)
        last_index = int(self.total_chunk_num - self.video_chunk_remain -1)
        future_chunk_length = self.mpc_horizon
        if (self.total_chunk_num - last_index < self.mpc_horizon ):
            future_chunk_length = self.total_chunk_num - last_index

        # planning for the optimal choice for next chunk
        # opt_a = solving_opt(self.env, self.start_buffer, self.last_bit_rate, future_chunk_length, self.rebuf_p, self.smooth_p)
        opt_a = self.env.solving_opt(
                                self.start_buffer, 
                                self.last_bit_rate, 
                                future_chunk_length, 
                                a_dim = 6, 
                                args = args
        )
        return opt_a

    def clear_buffer(self):
        self.past_errors = []
        self.past_bandwidth_ests = []

    def opt_action_robustMPC(self, state, args):
        # compute the optimal choice for next video chunk
        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(self.past_bandwidth_ests) > 0 ):
            curr_error  = abs(self.past_bandwidth_ests[-1]-state[0,-1])/float(state[0,-1])
        self.past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[0,-5:]
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
        if ( len(self.past_errors) < 5 ):
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        # past_bandwidth_ests.append(harmonic_bandwidth)
        self.past_bandwidth_ests.append(future_bandwidth)


        # future chunks length (try 4 if that many remaining)
        last_index = int(self.total_chunk_num - self.video_chunk_remain - 1)
        future_chunk_length = self.mpc_horizon
        if (self.total_chunk_num - last_index - 1 < self.mpc_horizon ):
            future_chunk_length = self.total_chunk_num - last_index - 1

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = self.start_buffer
        #start = time.time()
        for full_combo in self.CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_bit_rate_ = self.last_bit_rate
            for position in range(0, len(combo)): 
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                chunk_action = combo[position]
                chunk_quality_ = np.log(self.bitrate_versions[chunk_action] / \
                                    float(self.bitrate_versions[0])) if args.log else \
                                        self.bitrate_versions[chunk_action] / M_IN_K
                download_time = (self.env.video_size[chunk_action][index]/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += chunk_quality_
                last_quality_ = np.log(self.bitrate_versions[last_bit_rate_] / \
                                    float(self.bitrate_versions[0])) if args.log else \
                                        self.bitrate_versions[last_bit_rate_] / M_IN_K
                smoothness_diffs += abs(chunk_quality_ - last_quality_)
                last_bit_rate_ = chunk_action
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
            
            reward = bitrate_sum - (self.rebuffer_penalty*curr_rebuffer_time) \
                                    - self.smooth_penalty * smoothness_diffs

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

        return send_data

    def step(self, action): # execute the action 
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_chunk_sizes, \
                 end_of_video, video_chunk_remain, curr_chunk_sizes\
                    = self.env.get_video_chunk(action)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        self.last_bit_rate = self.bit_rate
        self.bit_rate = action
        self.start_buffer = buffer_size

        self.video_chunk_remain = video_chunk_remain

        if end_of_video:
            self.time_stamp = 0
            self.last_bit_rate = DEFAULT_QUALITY

        return delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
                next_chunk_sizes, end_of_video, video_chunk_remain, curr_chunk_sizes

