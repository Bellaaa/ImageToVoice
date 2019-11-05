from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import cv2

def corp_convert(path, start, end):
	try:
		ffmpeg_extract_subclip('VoxCeleb1/'+path+'.mp4', start, end, targetname='VoxCeleb1/'+path+'tmp'+'.mp4')
	except IOError as err:
		return
    # Convert video into .wav file
	os.system('ffmpeg -i {from_path}{ext} {to_path}{name}.wav'.format(ext='.mp4', from_path='VoxCeleb1/'+path+'tmp', to_path='VoxCeleb1/face_audio/', name=path))
	print('"{}" successfully converted into wav!'.format(path))

	# Image part
	vidcap = cv2.VideoCapture('VoxCeleb1/'+path+'tmp'+'.mp4')
	success, image = vidcap.read()
	cv2.imwrite("VoxCeleb1/face_image/%s.jpg" % path, image)

	os.remove("VoxCeleb1/{filename}tmp.mp4".format(filename=path))
