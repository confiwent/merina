import os 

BITRATE = [300,750,1200,1850,2850,4300]#kbps
video_path = './videos/'

for bitrate in range(len(BITRATE)):       
    os.chdir(video_path) # cd path
    command = 'mkdir' + ' ' + 'br_' + str(bitrate) # creat folder
    os.system(command)
    in_video_path = 'br_' + str(bitrate)
    os.chdir(in_video_path)
    command = 'MP4Box -dash 4000 -frag 4000 -segment-name chunk_ ../Avengers_' + str(BITRATE[bitrate]) + 'Kbps.mp4'  # transfer mp4 to dash file, each segment(chunk) has a 0.1s duration
    os.system(command)
    os.chdir('../../')