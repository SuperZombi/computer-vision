import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# https://wil.yegelwel.com/subplots-with-row-titles/
def subplots_with_row_titles(nrows, ncols, row_titles=None, row_title_kw=None, subplot_kw=None, **fig_kw):
	if row_titles is not None and len(row_titles) != nrows:
		raise ValueError(f'If row_titles is specified, there must be one for each row. Got={row_titles}')
	if row_title_kw is None:
		row_title_kw = {}
	if subplot_kw is None:
		subplot_kw = {}
	
	fig, big_axes = plt.subplots(nrows, 1, **fig_kw)
	if nrows == 1: big_axes = [big_axes]

	for (row, big_ax) in enumerate(big_axes):
		if row_titles is not None:
			big_ax.set_title(str(row_titles[row]), **row_title_kw)
			big_ax.axis('off')
	
	axarr = np.empty((nrows, ncols), dtype='O')
	for row in range(nrows):
		for col in range(ncols):
			ax= fig.add_subplot(nrows, ncols, row*ncols+col+1, **subplot_kw)
			axarr[row, col] = ax
	return fig, axarr


def displayImages(images, titles, window_title=None):
	rows, cols = len(images), len(images[0])
	fig, axes = subplots_with_row_titles(rows, cols, figsize=(cols*2, rows*2), row_titles=titles)

	for i, row in enumerate(images):
		for j, img in enumerate(row):
			axes[i, j].axis('off')
			axes[i, j].imshow(img)

	if window_title:
		fig.canvas.manager.set_window_title(window_title)
	plt.tight_layout()
	plt.show()

def displayImagesDict(images, cols=1, window_title=None):
	rows = math.ceil(len(images) / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
	if rows == 1: axes = np.array([axes])
	if cols == 1: axes = np.array([[item] for item in axes])

	i, j = 0, 0
	for caption, img in images.items():
		axes[i, j].axis('off')
		axes[i, j].imshow(img)
		axes[i, j].set_title(caption)
		j+=1
		if j == cols:
			j = 0
			i+=1

	if window_title:
		fig.canvas.manager.set_window_title(window_title)
	plt.tight_layout()
	plt.show()



class ImageAnalyzer:
	def __init__(self, path):
		self.img = cv2.imread(path)

	@property
	def BGR(self):
		return self.img
	@property
	def RGB(self):
		return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
	@property
	def LAB(self):
		return cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
	@property
	def YCB(self):
		return cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
	@property
	def HSV(self):
		return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

	def create_mask(self, color, thresh):
		minColor = np.array([color[0] - thresh,
						   color[1] - thresh,
						   color[2] - thresh])
		maxColor = np.array([color[0] + thresh,
						   color[1] + thresh,
						   color[2] + thresh])
		return cv2.inRange(self.img, minColor, maxColor)

	def apply_mask(self, mask):
		return cv2.bitwise_and(self.img, self.img, mask=mask)


class ColorConverter(ImageAnalyzer):
	def __init__(self, bgr):
		self.img = np.uint8([[bgr]])
	@property
	def BGR(self): return super().BGR.flatten()
	@property
	def RGB(self): return super().RGB.flatten()
	@property
	def LAB(self): return super().LAB.flatten()
	@property
	def YCB(self): return super().YCB.flatten()
	@property
	def HSV(self): return super().HSV.flatten()


if __name__ == "__main__":
	brigth = ImageAnalyzer("cube-1.png")
	dark = ImageAnalyzer("cube-2.png")

	displayImages([
		[brigth.RGB, dark.RGB],
		[brigth.BGR, dark.BGR],
		[brigth.LAB, dark.LAB],
		[brigth.YCB, dark.YCB],
		[brigth.HSV, dark.HSV]
	], ["RGB", "BGR", "LAB", "YCB", "HSV"], window_title="Порівняння колірних моделей")


	targetColor = ColorConverter([40, 158, 16]) # bgr

	maskBGR = brigth.create_mask(targetColor.BGR, thresh=40)
	resultBGR = brigth.apply_mask(maskBGR)

	maskHSV = brigth.create_mask(targetColor.HSV, thresh=40)
	resultHSV = brigth.apply_mask(maskHSV)

	maskYCB = brigth.create_mask(targetColor.YCB, thresh=40)
	resultYCB = brigth.apply_mask(maskYCB)

	maskLAB = brigth.create_mask(targetColor.LAB, thresh=40)
	resultLAB = brigth.apply_mask(maskLAB)


	displayImagesDict({
		"BGR": resultBGR,
		"HSV": resultHSV,
		"YCB": resultYCB,
		"LAB": resultLAB
	}, cols=2, window_title="Маскування")
