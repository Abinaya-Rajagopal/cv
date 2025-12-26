import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\nagur\Desktop\cv lab\Screenshot 2025-12-26 165904.png")
id_kernel = np.array([[0, 0, 0], [0, 1, 0],
 [0, 0, 0]])
flt_img = cv2.filter2D(src=img, ddepth=-1, kernel=id_kernel) 
flt_img_rgb = cv2.cvtColor(flt_img, cv2.COLOR_BGR2RGB)
plt.imshow(flt_img) 
plt.title('Identity Filter Result')
plt.axis('off') # Hide the axes
plt.show()

#canny edge detection
import cv2, matplotlib.pyplot as plt
img = cv2.imread(r'C:\\Users\\nagur\\Desktop\\cv lab\\Screenshot 2025-12-26 165904.png')
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

#sobel edge detection
import cv2, matplotlib.pyplot as plt
img = cv2.imread(r'C:\\Users\\nagur\\Desktop\\cv lab\\Screenshot 2025-12-26 165904.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel = cv2.magnitude(
    cv2.Sobel(gray, cv2.CV_64F, 1, 0, 5),
    cv2.Sobel(gray, cv2.CV_64F, 0, 1, 5)
)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(sobel, cmap='gray')
plt.axis('off')
plt.show()

import cv2, matplotlib.pyplot as plt
img = cv2.imread(r'C:\Users\nagur\Desktop\cv lab\Screenshot 2025-12-26 165904.png', 0)
_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.imshow(th, cmap='gray')
plt.title('Thresholding')
plt.axis('off')
plt.show()

