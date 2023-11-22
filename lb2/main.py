import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import cv2
import os
from PIL import Image
import numpy as np
import math


class TextureSelector:
	def __init__(self, path):
		image = Image.open(path).convert('L')
		image.thumbnail(size=(800,800))
		self.img = np.array(image)
		self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
		self.selections = []

	@property
	def selected(self):
		return len(self.selections)

	def selectArea(self):
		roi = cv2.selectROI("Select the area", self.img)
		if roi != (0, 0, 0, 0):
			self.selections.append(roi)
		cv2.destroyWindow("Select the area")

	def imageCrop(self, x1, y1, x2, y2):
		return self.img[y1:y1+y2, x1:x1+x2]

	def computGLCM(self):
		resultX = []
		resultY = []
		for patch in self.selections:
			image = self.imageCrop(*patch)
			glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
			resultX.append(graycoprops(glcm, 'dissimilarity')[0, 0])
			resultY.append(graycoprops(glcm, 'correlation')[0, 0])
		return resultX, resultY

	def display(self):
		colors = self.colors.copy()
		fig = plt.figure(figsize=(8, 8))
		rows = math.ceil(self.selected / 2) + 1

		ax = fig.add_subplot(rows, 2, 1)
		ax.imshow(self.img, cmap=plt.cm.gray, vmin=0, vmax=255)
		for (x1, y1, x2, y2) in self.selections:
			ax.axis('off')
			ax.plot(x1+x2/2, y1+y2/2, colors.pop(0)[0]+'s')

		colors = self.colors.copy()
		ax = fig.add_subplot(rows, 2, 2)
		glcmX, glcmY = self.computGLCM()
		for x, y in zip(glcmX, glcmY):
			ax.plot(x, y, colors.pop(0)[0]+'o')
		ax.set_xlabel('GLCM Dissimilarity')
		ax.set_ylabel('GLCM Correlation')

		colors = self.colors.copy()
		for i, patch in enumerate(self.selections):
			ax = fig.add_subplot(rows, 2, i+3)
			ax.imshow(self.imageCrop(*patch), cmap=plt.cm.gray, vmin=0, vmax=255)
			ax.set_xticks([])
			ax.set_yticks([])
			color = colors.pop(0)
			for axis in ['top','bottom','left','right']:
				ax.spines[axis].set_linewidth(3)
				ax.spines[axis].set_color(color)

		fig.canvas.manager.set_window_title("Grey level co-occurrence matrix features")
		plt.tight_layout()
		plt.show()


selector = TextureSelector("1.jpg")
# selector.img = data.camera()
selector.selections = [(279, 543, 75, 64), (97, 415, 36, 40), (174, 195, 30, 27), (563, 320, 28, 24), (406, 655, 41, 34), (620, 250, 43, 40)]

selecting = True
while selecting:
	os.system("cls")
	print(f"SELECTED: {selector.selected} areas\n------------------")
	choise = input("  1   — Select area\n<ANY> — Continue\n")
	if choise == "1":
		selector.selectArea()
	else:
		selecting = False

print(selector.selections)
selector.display()
