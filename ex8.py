#BM
import cv2
import numpy as np

capL = cv2.VideoCapture(r"C:\Users\SEC22AM036\Videos\Screen Recordings\L1VIDEO.mp4")
capR = cv2.VideoCapture(r"C:\Users\SEC22AM036\Videos\Screen Recordings\L2VIDEO.mp4")

if not capL.isOpened() or not capR.isOpened():
    print("Error opening videos")
    exit()

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(grayL, grayR)
    disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp = np.uint8(disp)

    cv2.imshow("Left Frame", frameL)
    cv2.imshow("Right Frame", frameR)
    cv2.imshow("Disparity Map", disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()

#SGBM
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    print("Error opening video")
    exit()

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=80,
    blockSize=15,
    P1=8 * 3 * 15 * 15,
    P2=32 * 3 * 15 * 15
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    left = cv2.cvtColor(frame[:, :w//2], cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(frame[:, w//2:], cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp = cv2.resize(np.uint8(disp), (w, h))

    cv2.imshow("Disparity Map - SGBM", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()