import cv2
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.util import random_noise
from skimage.filters import median, gaussian
from skimage.morphology import disk


img1 = cv2.imread('medic-1.png', cv2.IMREAD_GRAYSCALE)
# img1 = rescale(img1, 0.15)

img_ns = random_noise(img1, mode='speckle', mean=0.1)

img_m3 = median(img_ns, disk(3))
img_m9 = median(img_ns, disk(9))

img_g1 = gaussian(img_ns, 1)
img_g3 = gaussian(img_ns, 3)


plt.subplot(3, 2, 1)
plt.imshow(img1, cmap='grey')
plt.axis('off')
plt.title("Оригінал")

plt.subplot(3, 2, 2)
plt.imshow(img_ns, cmap='grey')
plt.axis('off')
plt.title("Шумне зображення")

plt.subplot(3, 2, 3)
plt.imshow(img_m3, cmap='grey')
plt.axis('off')
plt.title("Медіана (3)")

plt.subplot(3, 2, 4)
plt.imshow(img_m9, cmap='grey')
plt.axis('off')
plt.title("Медіана (9)")

plt.subplot(3, 2, 5)
plt.imshow(img_g1, cmap='grey')
plt.axis('off')
plt.title("Гаус (1)")

plt.subplot(3, 2, 6)
plt.imshow(img_g3, cmap='grey')
plt.axis('off')
plt.title("Гаус (3)")

plt.tight_layout()
plt.show()
