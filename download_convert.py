import videocorp as vc
import sys
import os
import time
import csv

# import videocorp as vc
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# youtube-dl -o -"VoxCeleb1/%(title)s-%(id)s.%(ext)s" -f mp4 https://www.youtube.com/watch\?v\=1zcIwhmdeo4  

def download_video(csvfile):
    with open(csvfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            url, start, end = row[1], float(row[2])/25, float(row[3])/25
            print(start, end)
            download(url)
            try:
                vc.corp_convert('{filename}'.format(filename=url), start, end)
            except IOError as err:
                print('{filename} is not avaliable anymore'.format(filename=url))
                pass
            try:
                os.remove("VoxCeleb1/{filename}.mp4".format(filename=url))
            except  FileNotFoundError as err:
                pass

def download(path):
    print(path)

    # Convert video into .wav file
    # os.system('ffmpeg -i {from_path}{file}{ext} {to_path}{file}.wav'.format(file=file, ext=extension, from_path=datadir, to_path='face_audio/'))
    os.system('youtube-dl -o "VoxCeleb1/%(id)s.%(ext)s" -f mp4 {url}'.format(url=path))
    # Convert .wav into final .mp3 file
    # os.system('lame {file}.wav {file}.mp3'.format(file=file))
    # os.remove('{}.wav'.format(file))  # Deletes the .wav file
    print('"{}" successfully downloaded!'.format(path))



download_video('ids.csv')