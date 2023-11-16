import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('medic-1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('medic-2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.subplot(2, 2, 1)
plt.imshow(img1)
plt.axis('off')
plt.title('Image 1')

plt.subplot(2, 2, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('Image 2')

plt.subplot(2, 2, 3)
plt.hist(img1.ravel(), 256, (0, 256))
plt.xlabel('Intensity, (0..255)')
plt.ylabel('Count, px')

plt.subplot(2, 2, 4)
plt.hist(img2.ravel(), 256, (0, 256))
plt.xlabel('Intensity, (0..255)')
plt.ylabel('Count, px')

plt.tight_layout()
plt.show()
