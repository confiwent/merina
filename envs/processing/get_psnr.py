import os 
import re

BITRATE = [300,750,1200,1850,2850,4300]#kbps
CHUNK_DURATION = 4 # sec
FPS = 60 # frame per seconds
TOTAL_CHUNKS = 150
FILE_PATH = './'

### calculate the psnr per frame

for level in range(len(BITRATE)):
    command = 'ffmpeg -i ' + './videos/Avengers_' + str(BITRATE[level]) + 'Kbps.mp4 -i '+ './Avengers.mp4' + ' -lavfi psnr="stats_file=psnr_' + str(level) + '.log" -f null -'
    os.system(command)  ## psnr


### calculate the psnr and ssim per CHUNK

for level in range(len(BITRATE)):
    psnr_per_frame = [] # record the psnr 
    psnr_per_chunk = []
    with open(FILE_PATH + 'psnr_' + str(level) + '.log', 'r') as pf:
        for line in pf:
            parse = line.split()
            if len(parse) <= 1:
                break
            psnr = re.findall(r"\d+\.?\d*", parse[5]) ## extract the average psnr per frame
            if len(psnr):
                psnr_per_frame.append(float(psnr[0])) # record the average psnr
            else:
                psnr_per_frame.append(0)

    for chunk in range(TOTAL_CHUNKS):
        chunk_psnr_sum = 0
        valid_f = 0 ## valif frame (eliminate the frame where the psnr is inf)
        for fra_c in range(CHUNK_DURATION * FPS):
            fra_g = chunk * CHUNK_DURATION * FPS + fra_c
            if psnr_per_frame[fra_g]:
                valid_f +=1
                chunk_psnr_sum += psnr_per_frame[fra_g]
        psnr_per_chunk.append(chunk_psnr_sum/valid_f) 

    with open('chunk_psnr' + str(level), 'w') as pf:
        assert len(psnr_per_chunk) == TOTAL_CHUNKS
        for chunk in range(TOTAL_CHUNKS):
            pf.write(str(chunk) + '\t' + str(psnr_per_chunk[chunk]) + '\n')
