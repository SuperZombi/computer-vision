import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d

def lucas_kanade(imfirst, imsecond):
	Image1 = cv.cvtColor(imfirst, cv.COLOR_BGR2GRAY)
	Image2 = cv.cvtColor(imsecond, cv.COLOR_BGR2GRAY)

	color = np.random.randint(0, 255, (100,3))
	Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))
	Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))
	Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))
	Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))

	lx = (convolve2d(Image1, Gx) + convolve2d(Image2, Gx)) / 2
	ly = (convolve2d(Image1, Gy) + convolve2d(Image2, Gy)) / 2
	lt1 = convolve2d(Image1, Gt1) + convolve2d(Image2, Gt2)

	feature = cv.goodFeaturesToTrack(Image1, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
	feature = np.reshape(feature, newshape=[-1, 2])

	u = np.ones(lx.shape)
	v = np.ones(lx.shape)
	status = np.zeros(feature.shape[0])
	A = np.zeros((2, 2))
	B = np.zeros((2, 1))
	mask = np.zeros_like(imfirst)

	newFeature = np.zeros_like(feature)

	for a, b in enumerate(feature):
		x, y = int(b[0]), int(b[1])
		A[0, 0] = np.sum((lx[y-1:y+2, x-1:x+2]) ** 2)
		A[1, 1] = np.sum((lx[y-1:y+2, x-1:x+2]) ** 2)
		A[0, 1] = np.sum(lx[y-1:y+2, x-1:x+2] * ly[y-1:y+2, x-1:x+2])
		A[1, 0] = np.sum(lx[y-1:y+2, x-1:x+2] * ly[y-1:y+2, x-1:x+2])
		Ainv = np.linalg.pinv(A)

		B[0, 0] = -np.sum(lx[y-1:y+2, x-1:x+2] * lt1[y-1:y+2, x-1:x+2])
		B[1, 0] = -np.sum(lx[y-1:y+2, x-1:x+2] * lt1[y-1:y+2, x-1:x+2])
		prod = np.matmul(Ainv, B)

		u[y, x] = prod[0].item()
		v[y, x] = prod[1].item()

		newFeature[a] = [np.int32(x+u[y,x]), np.int32(y+v[y,x])]
		if np.int32(x+u[y,x]) == x and np.int32(y+v[y,x]) == y:
			status[a] = 0
		else:
			status[a] = 1

	um = np.flipud(u)
	vm = np.flipud(v)

	good_new = newFeature[status==1]
	good_old = feature[status==1]

	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a,b = new.ravel()
		c,d=old.ravel()
		a,b,c,d = map(int, [a,b,c,d])
		mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		imsecond = cv.circle(imsecond, (a,b), 5, color[i].tolist(), -1)

	img = cv.add(imsecond, mask)
	return img


cap = cv.VideoCapture("video.mp4")
old_frame = []

while cap.isOpened():
	ret, img = cap.read()
	if not ret: break

	if len(old_frame) > 0:
		image = lucas_kanade(old_frame, img)
		cv.imshow('frame',image)
		k = cv.waitKey(30)
		if k == 27:
			break

	old_frame = img

cv.destroyAllWindows()
cap.release()
