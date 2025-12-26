#horn
import cv2, numpy as np

cap = cv2.VideoCapture(r"C:\Users\nagur\Desktop\cv lab\video.mp4")

ret, old = cap.read()
if not ret:
    print("Error: Video not opened")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(old)
hsv[...,1] = 255

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Dense Optical Flow",
               np.hstack((frame, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))))

    if cv2.waitKey(30) & 0xFF == 27:
        break

    old_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()