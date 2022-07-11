import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
# NOISE_LOW = 0.9
# NOISE_HIGH = 1.1
# VIDEO_SIZE_FILE = '/size/video_size_'


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, video_size_file, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(video_size_file + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

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

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain

    def get_download_time(self,trace_idx, video_chunk_counter, mahimahi_ptr,last_mahimahi_time, chunk_quality):
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
                    last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp-1]
                duration_all += duration
            download_time.append(duration_all)
            if quality == chunk_quality:
                trace_idx_ = trace_idx
                video_chunk_counter_ = video_chunk_counter
                mahimahi_ptr_ = mahimahi_ptr_tmp
                last_mahimahi_time_ = last_mahimahi_time_tmp

        ## -------------------- test whether end of video ---------------------------------------------------
        video_chunk_counter_ += 1
        if video_chunk_counter_ >= TOTAL_VIDEO_CHUNCK:

            video_chunk_counter_ = 0
            # trace_idx_ += 1
            # if trace_idx_ >= len(self.all_cooked_time):
            #     trace_idx_ = 0

            # cooked_time = self.all_cooked_time[trace_idx_]
            # cooked_bw = self.all_cooked_bw[trace_idx_]

            # randomize the start point of the video
            # note: trace file starts with time 0
            mahimahi_ptr_ = 1
            last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


        if len(download_time)==1:
            return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
        else:
            return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_