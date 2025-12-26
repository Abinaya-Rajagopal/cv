#semantic segmentation
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
img = img_as_float(data.camera())
ls = morphological_chan_vese(img, 35, checkerboard_level_set(img.shape,6), 3)
plt.imshow(img, cmap='gray'); plt.axis('off'); plt.title("Original Image"); plt.show()
plt.imshow(img, cmap='gray')
plt.contour(ls, [0.5], colors='r')
plt.axis('off')
plt.show()

#instance segmentation
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
img = img_as_float(data.camera())
ls = morphological_chan_vese(img, 35, checkerboard_level_set(img.shape,6), 3)
plt.imshow(ls, cmap='gray')
plt.axis('off')
plt.title("Instance Segmentation")
plt.show()
