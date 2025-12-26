#lucas
import cv2, numpy as np

cap = cv2.VideoCapture(r"C:\Users\nagur\Desktop\cv lab\video.mp4")

ret, old = cap.read()
old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, 100, 0.3, 7)
mask = np.zeros_like(old)
colors = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None)

    for i, (n, o) in enumerate(zip(p1[st==1], p0[st==1])):
        a,b = n.ravel()
        c,d = o.ravel()
        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a),int(b)), 4, colors[i].tolist(), -1)

    cv2.imshow("Lucas Kanade", cv2.add(frame, mask))
    if cv2.waitKey(30) & 0xFF == 27:
        break

    old_gray = gray.copy()
    p0 = p1[st==1].reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()

