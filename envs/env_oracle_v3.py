''' 
A version for logarithmic or linear QoE metric which ignores

the perceptual video quality

'''

import numpy as np


MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
M_IN_K = 1000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1

# pythran export class Environment()
class Environment:
    # pythran export __init__(float list, float list, str, str, int)
    def __init__(
            self, 
            all_cooked_time, 
            all_cooked_bw, 
            video_size_file,
            random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        
        self.s_info = 17
        self.s_len = 10
        self.c_len = 3
        self.bitrate_version = [300, 750, 1200, 1850, 2850, 4300]
        self.br_dim = len(self.bitrate_version)
        self.qual_p = 0.85
        self.rebuff_p = 4.3
        self.smooth_p = 1
        self.smooth_n = 0.3

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.br_dim):
            self.video_size[bitrate] = []
            with open(video_size_file + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.total_chunk_num = len(self.video_size[0])
        self.chunk_length_max = self.total_chunk_num

    # pythran export set_env_info(int, int, int, int, int list, float, float)
    def set_env_info(
            self, 
            s_info, 
            s_len, 
            c_len, 
            chunk_num, 
            br_version, 
            qual_p, 
            rebuff_p, 
            smooth_p,
            smooth_n
    ):
        self.s_info = s_info
        self.s_len = s_len
        self.c_len = c_len
        self.total_chunk_num = chunk_num
        self.bitrate_version = br_version
        self.br_dim = len(self.bitrate_version)
        self.qual_p = qual_p
        self.rebuff_p = rebuff_p
        self.smooth_p = smooth_p
        self.smooth_n = smooth_n

    # pythran export get_env_info(None)
    def get_env_info(self):
        return self.s_info, self.s_len , self.c_len, \
                self.total_chunk_num, self.bitrate_version, \
                self.rebuff_p, self.smooth_p

    # pythran export get_video_chunk(int)
    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.br_dim

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        curr_chunk_sizes = []
        for i in range(self.br_dim):
            curr_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_chunk_num - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_chunk_num:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(self.br_dim):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain, \
            curr_chunk_sizes

    # pythran export get_download_time_upward(int, int, int, float, int)
    def get_download_time_upward(
                            self, 
                            trace_idx, 
                            video_chunk_counter, 
                            mahimahi_ptr,
                            last_mahimahi_time, 
                            chunk_quality
    ):
        ## ---------------- compute last time ----------------------------------------------------
        if trace_idx == -1:
            trace_idx = self.trace_idx
            video_chunk_counter = self.video_chunk_counter
            mahimahi_ptr = self.mahimahi_ptr
            cooked_time = self.all_cooked_time[trace_idx]
            last_mahimahi_time = self.last_mahimahi_time
        ## ----------------- assign values ----------------------------------------------------

        cooked_bw = self.all_cooked_bw[trace_idx]
        cooked_time = self.all_cooked_time[trace_idx]

        ## ------------------- compute true bandwidth --------------------------------------------
        download_time = []
        for quality in range(chunk_quality, min(chunk_quality + 2, 6)):
            duration_all = 0
            video_chunk_counter_sent = 0  # in bytes
            video_chunk_size = self.video_size[quality][video_chunk_counter]
            mahimahi_ptr_tmp = mahimahi_ptr
            last_mahimahi_time_tmp = last_mahimahi_time

            while True:  # download video chunk over mahimahi
                throughput = cooked_bw[mahimahi_ptr_tmp] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = cooked_time[mahimahi_ptr_tmp] \
                           - last_mahimahi_time_tmp

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    last_mahimahi_time_tmp += fractional_time
                    duration_all += fractional_time
                    break
                video_chunk_counter_sent += packet_payload
                last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp]
                mahimahi_ptr_tmp += 1

                if mahimahi_ptr_tmp >= len(cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    mahimahi_ptr_tmp = 1
                    last_mahimahi_time_tmp = 0
                duration_all += duration
            download_time.append(duration_all)
            if quality == chunk_quality:
                trace_idx_ = trace_idx
                video_chunk_counter_ = video_chunk_counter
                mahimahi_ptr_ = mahimahi_ptr_tmp
                last_mahimahi_time_ = last_mahimahi_time_tmp

        ## -------------------- test whether end of video ---------------------------------------------------
        video_chunk_counter_ += 1
        if video_chunk_counter_ >= self.total_chunk_num:

            video_chunk_counter_ = 0
            trace_idx_ += 1
            if trace_idx_ >= len(self.all_cooked_time):
                trace_idx_ = 0

            cooked_time = self.all_cooked_time[trace_idx_]
            cooked_bw = self.all_cooked_bw[trace_idx_]

            # randomize the start point of the video
            # note: trace file starts with time 0
            mahimahi_ptr_ = self.mahimahi_start_ptr
            last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


        if len(download_time)==1:
            return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
        else:
            return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_

    # pythran export get_download_time_downward(int, int, int, float, int)
    def get_download_time_downward(
                            self,
                            trace_idx, 
                            video_chunk_counter, 
                            mahimahi_ptr,
                            last_mahimahi_time, 
                            chunk_quality
    ):
        ## ---------------- compute last time ----------------------------------------------------
        if trace_idx == -1:
            trace_idx = self.trace_idx
            video_chunk_counter = self.video_chunk_counter
            mahimahi_ptr = self.mahimahi_ptr
            cooked_time = self.all_cooked_time[trace_idx]
            last_mahimahi_time = self.last_mahimahi_time
        ## ----------------- assign values ----------------------------------------------------

        cooked_bw = self.all_cooked_bw[trace_idx]
        cooked_time = self.all_cooked_time[trace_idx]

        ## ------------------- compute true bandwidth --------------------------------------------
        download_time = []
        for quality in range(chunk_quality, max(chunk_quality - 2, -1), -1):
            duration_all = 0
            video_chunk_counter_sent = 0  # in bytes
            video_chunk_size = self.video_size[quality][video_chunk_counter]
            mahimahi_ptr_tmp = mahimahi_ptr
            last_mahimahi_time_tmp = last_mahimahi_time

            while True:  # download video chunk over mahimahi
                throughput = cooked_bw[mahimahi_ptr_tmp] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = cooked_time[mahimahi_ptr_tmp] \
                           - last_mahimahi_time_tmp

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    last_mahimahi_time_tmp += fractional_time
                    duration_all += fractional_time
                    break
                video_chunk_counter_sent += packet_payload
                last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp]
                mahimahi_ptr_tmp += 1

                if mahimahi_ptr_tmp >= len(cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    mahimahi_ptr_tmp = 1
                    last_mahimahi_time_tmp = 0
                duration_all += duration
            download_time.append(duration_all)
            if quality == chunk_quality:
                trace_idx_ = trace_idx
                video_chunk_counter_ = video_chunk_counter
                mahimahi_ptr_ = mahimahi_ptr_tmp
                last_mahimahi_time_ = last_mahimahi_time_tmp

        ## -------------------- test whether end of video ---------------------------------------------------
        video_chunk_counter_ += 1
        if video_chunk_counter_ >= self.total_chunk_num:

            video_chunk_counter_ = 0
            trace_idx_ += 1
            if trace_idx_ >= len(self.all_cooked_time):
                trace_idx_ = 0

            cooked_time = self.all_cooked_time[trace_idx_]
            cooked_bw = self.all_cooked_bw[trace_idx_]

            # randomize the start point of the video
            # note: trace file starts with time 0
            mahimahi_ptr_ = self.mahimahi_start_ptr
            last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


        if len(download_time)==1:
            return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
        else:
            return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_

    def get_quality(self, log, bitrate):
        chunk_quality = np.log(self.bitrate_version[bitrate] / \
                                    float(self.bitrate_version[0])) if log else \
                                        self.bitrate_version[bitrate] / M_IN_K
        return chunk_quality

    # pythran export searching_upward(float, int, int, float, float, int)
    def searching_upward(self, 
                        start_buffer, 
                        last_bit_rate, 
                        future_chunk_length,
                        args
    ):
        # load the QoE hyperparameters
        qual_p = self.qual_p
        rebuf_penalty = self.rebuff_p
        smooth_p = self.smooth_p
        a_dim = self.br_dim

        max_reward = -10000000000
        reward_comparison = False
        send_data = 0
        parents_pool = [[0.0,-1,-1,-1,-1, start_buffer, int(last_bit_rate)]]
        for position in range(future_chunk_length):
            if position == future_chunk_length-1:
                reward_comparison = True
            children_pool = []
            for parent in parents_pool:
                action = -1 
                High_Maybe_Superior = True

                while High_Maybe_Superior:
                    trace_idx = parent[1]
                    video_chunk_counter = parent[2]
                    mahimahi_ptr = parent[3]
                    last_mahimahi_time = parent[4]
                    curr_buffer = parent[5]
                    last_quality = parent[-1] # the index must be -1
                    curr_rebuffer_time = 0
                    chunk_quality = action + 1
                    # get download time with true bandwidth
                    download_time, download_time_upward, trace_idx_, \
                        video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ \
                            = self.get_download_time_upward(
                                            trace_idx, 
                                            video_chunk_counter, 
                                            mahimahi_ptr, 
                                            last_mahimahi_time, 
                                            chunk_quality
                                )

                    # simulate the video playback, including rebuffering events
                    if (curr_buffer < download_time):
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0.0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4

                    # record the instant reward for each video chunk 
                    curr_chunk_quality_ = self.get_quality(args.log, chunk_quality)
                    last_chunk_quality_ = self.get_quality(args.log, last_quality)
                    smoothness_diffs = abs(curr_chunk_quality_ - last_chunk_quality_)

                    ## reward = video quality - rebuffering penality - smoothness penality
                    reward = qual_p * curr_chunk_quality_ \
                                - rebuf_penalty * curr_rebuffer_time \
                                    - smooth_p * smoothness_diffs
                    reward += parent[0]

                    # children node
                    children = parent[:]
                    children[0] = reward
                    children[1] = trace_idx_
                    children[2] = video_chunk_counter_
                    children[3] = mahimahi_ptr_
                    children[4] = last_mahimahi_time_
                    children[5] = curr_buffer
                    children.append(chunk_quality)
                    children_pool.append(children)

                    # update of maximum rewards 
                    if (reward >= max_reward) and reward_comparison:
                        if send_data > children[7] and reward == max_reward:
                            send_data = send_data
                        else:
                            send_data = children[7]  # index must be 7, not 6 or -1, since the length of children's list will increase with the tree being expanded
                        max_reward = reward

                    action += 1
                    if action + 1 == a_dim:
                        break

                    # criterion terms which have not been publicated
                    # theta = 
                    # parent[5] is the value of current buffer occupancy
                    rebuffer_term = rebuf_penalty * (max(download_time_upward - parent[5], 0) \
                                            - max(download_time - parent[5], 0))
                    q_a = self.get_quality(args.log, action)
                    q_a_ = self.get_quality(args.log, action + 1)

                    # here we determine whether it is worthwhile to enhance the bitrate level 
                    High_Maybe_Superior = ((1.0 + 2 * smooth_p) * (q_a - q_a_) \
                                            + rebuffer_term < 0.0) if (action + 1 <= parent[-1]) else \
                                                ((q_a - q_a_) + rebuffer_term < 0.0)

            parents_pool = children_pool

        return send_data

    # pythran export searching_downward(float, int, int, float, float, int)
    def searching_downward(self, 
                        start_buffer, 
                        last_bit_rate, 
                        future_chunk_length,
                        args
    ):
        # load the QoE hyperparameters
        qual_p = self.qual_p
        rebuf_penalty = self.rebuff_p
        smooth_p = self.smooth_p
        a_dim = self.br_dim

        max_reward = -10000000000
        reward_comparison = False
        send_data = 0
        parents_pool = [[0.0,-1,-1,-1,-1, start_buffer, int(last_bit_rate)]]
        for position in range(future_chunk_length):
            if position == future_chunk_length-1:
                reward_comparison = True
            children_pool = []
            for parent in parents_pool:
                action = int(a_dim)
                Low_is_Superior = True
                
                while Low_is_Superior:
                    trace_idx = parent[1]
                    video_chunk_counter = parent[2]
                    mahimahi_ptr = parent[3]
                    last_mahimahi_time = parent[4]
                    curr_buffer = parent[5]
                    last_quality = parent[-1]
                    curr_rebuffer_time = 0
                    chunk_quality = action - 1
                    # get download time with true bandwidth
                    download_time, download_time_downward, trace_idx_, \
                        video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ \
                            = self.get_download_time_downward(trace_idx, video_chunk_counter, \
                                                        mahimahi_ptr, last_mahimahi_time, chunk_quality)

                    # simulate the video playback, including rebuffering events
                    if (curr_buffer < download_time):
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0.0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4

                    # record the instant reward for each video chunk 
                    curr_chunk_quality_ = self.get_quality(args.log, chunk_quality)
                    last_chunk_quality_ = self.get_quality(args.log, last_quality)
                    smoothness_diffs = abs(curr_chunk_quality_ - last_chunk_quality_)

                    ## reward = video quality - rebuffering penality - smoothness penality
                    reward = qual_p * curr_chunk_quality_ \
                                - rebuf_penalty * curr_rebuffer_time \
                                    - smooth_p * smoothness_diffs
                    reward += parent[0]

                    # children node
                    children = parent[:]
                    children[0] = reward
                    children[1] = trace_idx_
                    children[2] = video_chunk_counter_
                    children[3] = mahimahi_ptr_
                    children[4] = last_mahimahi_time_
                    children[5] = curr_buffer
                    children.append(chunk_quality)
                    children_pool.append(children)

                    # update of maximum rewards 
                    if (reward >= max_reward) and reward_comparison:
                        if send_data > children[7] and reward == max_reward:
                            send_data = send_data
                        else:
                            send_data = children[7]  # index must be 7, not 6 or -1, since the length of children's list will increase with the tree being expanded
                        max_reward = reward

                    action -= 1
                    if action == 0:
                        break

                    # criterion terms
                    # theta = 
                    rebuffer_term = rebuf_penalty * (max(download_time - parent[5], 0) \
                                                - max(download_time_downward - parent[5], 0))
                    q_a = self.get_quality(args.log, action)
                    q_a_ = self.get_quality(args.log, action - 1)

                    Low_is_Superior = ((1.0 + 2 * smooth_p) * (q_a_ - q_a) + rebuffer_term >= 0.0) \
                                        if (action <= parent[-1]) else ((q_a_ - q_a) + rebuffer_term >= 0.0)

            parents_pool = children_pool
        return send_data

    # pythran export solving_opt(float, int, int, float, float, int)
    def solving_opt(self, start_buffer, last_bit_rate, future_chunk_length, a_dim, args):
        # opt_solution = self.searching_upward(
        #                                 start_buffer, 
        #                                 last_bit_rate, 
        #                                 future_chunk_length,
        #                                 args
        #                                 )
        if last_bit_rate < 0.5 * a_dim:
            opt_solution = self.searching_upward(
                                            start_buffer, 
                                            last_bit_rate, 
                                            future_chunk_length,
                                            args
                                            )
        else:
            opt_solution = self.searching_downward(
                                            start_buffer, 
                                            last_bit_rate, 
                                            future_chunk_length,
                                            args
                                            )
        return opt_solution