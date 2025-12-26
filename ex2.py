#harris corner
import cv2, numpy as np, matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\nagur\Desktop\cv lab\Screenshot 2025-12-26 175012.png")
h = cv2.cornerHarris(np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),2,5,0.07)
img[h > 0.01*h.max()] = [0,0,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

#SIFT
import cv2
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\nagur\Desktop\cv lab\Screenshot 2025-12-26 175012.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints_sift, _ = sift.detectAndCompute(gray_image, None)
sift_image = cv2.drawKeypoints(
    image, keypoints_sift, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

#surf
import cv2, matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\nagur\Desktop\cv lab\Screenshot 2025-12-26 175012.png")
kp = cv2.ORB_create().detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)[0]
out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


