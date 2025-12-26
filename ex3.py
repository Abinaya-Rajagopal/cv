


#haar cascade
import cv2
face = cv2.CascadeClassifier(cv2.data.haarcascades +
                             'haarcascade_frontalface_default.xml')
img = cv2.imread(r'C:\Users\nagur\Desktop\cv lab\croud.jpeg')
if img is None:
    print("Image not found")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
cv2.imwrite('ex3_harr_out.jpg', img)
print("Output saved as ex3_harr_out.jpg")