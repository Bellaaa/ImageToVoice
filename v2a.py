import sys
import os
import time
 
 
def video_to_mp3(datadir):
    """ Transforms video file into a MP3 file """
    try:
        for root, directories, imagenames in os.walk(datadir):
            for imagename in imagenames:
                file, extension = os.path.splitext(imagename)
                print(file)
                # Convert video into .wav file
                os.system('ffmpeg -i {from_path}{file}{ext} {to_path}{file}.wav'.format(file=file, ext=extension, from_path=datadir, to_path='face_audio/'))
                # Convert .wav into final .mp3 file
                # os.system('lame {file}.wav {file}.mp3'.format(file=file))
                # os.remove('{}.wav'.format(file))  # Deletes the .wav file
                print('"{}" successfully converted into wav!'.format(file))
    except OSError as err:
        print(err.reason)
        exit(1)
 
 
def main():
    # Confirm the script is called with the required params
    if len(sys.argv) != 2:
        print('Usage: python video_to_mp3.py FILE_NAME')
        exit(1)
 
    file_path = sys.argv[1]
    try:
        if not os.path.exists(file_path):
            print('file "{}" not found!'.format(file_path))
            exit(1)
 
    except OSError as err:
        print(err.reason)
        exit(1)
 
    video_to_mp3(file_path)
    time.sleep(1)
 
 
if __name__ == '__main__':
    main()