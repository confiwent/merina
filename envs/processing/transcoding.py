import os

filepath = 'Avengers.mp4'  # path of files to be handle
resultpath = './versions/'

BITRATE = [300,750,1200,1850,2850,4300] # Kbps

# os.chdir(resultpath)  # cd the object folder

for sample in range(len(BITRATE)): #len(BITRATE)
    # command =  'ffmpeg -i ' + filepath + ' -c:v libx265 -b:v ' + str(BITRATE[sample]) \
    # + 'k ' + '-r 60 -s 3840x1600 ' + resultpath + 'Avengers_' + str(BITRATE[sample]) + 'kbps.mp4'  # h265
    command =  'ffmpeg -i ' + filepath + ' -c:v libvpx-vp9 -b:v ' + str(BITRATE[sample]) \
    + 'k ' + '-r 60 -s 3840x1600 ' + resultpath + 'Avengers_' + str(BITRATE[sample]) + 'kbps.mp4'  # h265
    # transcode the video in different bitrate 
    # Notice: it not transfrom the frame-rate here !!
    # command = 'kvazaar -i ' + filepath + ' --input-res 3840x1600 --input-fps ' \
    #      + str(60) + ' --bitrate ' + str(BITRATE[sample]) + ' --output ' + resultpath + 'Avengers_' + str(BITRATE[sample]) +'kbps.mp4'
    os.system(command)  # execute the command above in system cmd