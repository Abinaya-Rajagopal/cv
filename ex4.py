import cv2
cap = cv2.VideoCapture(r"C:\Users\nagur\Desktop\cv lab\video.mp4")
ret, frame = cap.read()
tracker = cv2.legacy.TrackerKCF_create()
bbox = cv2.selectROI("Select", frame, False)
tracker.init(frame, bbox)
while True:
    ret, frame = cap.read()
    if not ret: break

    ok, bbox = tracker.update(frame)
    if ok:
        x,y,w,h = map(int, bbox)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    else:
        cv2.putText(frame, "Fail", (40,40), 0, 1, (0,0,255), 2)

    cv2.imshow("KCF", frame)
    if cv2.waitKey(30) & 0xFF == 27: break
cap.release()
cv2.destroyAllWindows()


#MOSSE
import cv2
t = cv2.legacy.TrackerMOSSE_create()
v = cv2.VideoCapture(r"C:\Users\nagur\Desktop\cv lab\video.mp4")
_, f = v.read()
b = cv2.selectROI("Select", f)
t.init(f, b)
while True:
    _, f = v.read()
    if f is None: break
    ok, b = t.update(f)
    if ok:
        x,y,w,h = map(int,b)
        cv2.rectangle(f,(x,y),(x+w,y+h),(255,0,0),2)
    else:
        cv2.putText(f,"Fail",(50,50),0,1,(0,0,255),2)
    cv2.imshow("MOSSE", f)
    if cv2.waitKey(1)==27: break
v.release()
cv2.destroyAllWindows()