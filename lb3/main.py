import cv2
import matplotlib.pyplot as plt
import os
from moviepy.editor import VideoFileClip

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
	return image, len(faces) > 0

def single_photo(file):
	img = cv2.imread(file)
	img, result = detect(img)
	if not result:
		print("Face not found")
	else:
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		plt.axis('off')
		plt.tight_layout()
		fig = plt.gcf()
		fig.canvas.manager.set_window_title("Face detected")
		plt.show()

def video(inputfile, outputfile, tempfile='temp.mp4'):
	cap = cv2.VideoCapture(inputfile)
	fps = cap.get(cv2.CAP_PROP_FPS)
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	out = cv2.VideoWriter(tempfile, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	current_frame = 0

	while cap.isOpened():
		ret, img = cap.read()
		if not ret: break

		print(f"{current_frame}/{total_frames}", end="\r")
		img, result = detect(img)
		out.write(img)
		current_frame+=1

	cap.release()
	out.release()

	# Склейка аудіо
	video_with_audio = VideoFileClip(inputfile)
	video_without_audio = VideoFileClip(tempfile, audio=False)
	video_without_audio = video_without_audio.set_audio(video_with_audio.audio)
	video_without_audio.write_videofile(outputfile)
	video_with_audio.close()
	video_without_audio.close()

	os.remove(tempfile)

single_photo('photo_1.jpg')
# video('1.mp4', 'output.mp4')