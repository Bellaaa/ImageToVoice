import cv2
import os

# https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/

for root, directories, imagenames in os.walk('facevoice_speech_dataset'):
	for imagename in imagenames:
		file, extension = os.path.splitext(imagename)
		vidcap = cv2.VideoCapture('facevoice_speech_dataset/' + imagename)
		success, image = vidcap.read()
		cv2.imwrite("face_image/%s.jpg" % file, image)
