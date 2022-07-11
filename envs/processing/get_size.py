import os 

TOTAL_VIDEO_CHUNCK = 150
BITRATE_LEVELS = 6
VIDEO_PATH = '../../videos/br_'
SIZE_DIR = './videos_info/videosize'

# assume videos are in ../video_servers/video[0, 1, 2, 3, 4, 5]
# the quality at video_0 is the lowest and video_5 is the highest

if not os.path.exists(SIZE_DIR):
      os.mkdir(SIZE_DIR)

for bitrate in range(BITRATE_LEVELS):
    os.chdir(SIZE_DIR)
    with open('video_size_' + str(bitrate), 'w') as f:
        for chunk_num in range(1, TOTAL_VIDEO_CHUNCK + 1):
            video_chunk_path = VIDEO_PATH + str(bitrate) + '/' + \
                                            'chunk_' + str(chunk_num) + '.m4s'
            chunk_size = os.path.getsize(video_chunk_path)
            f.write(str(chunk_size) + '\n')
    os.chdir('../../')