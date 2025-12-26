#roboflow
import os
from roboflow import Roboflow
from PIL import Image
rf = Roboflow(api_key="IBj2SprpUGGuAPFbNDk0")
workspace = rf.workspace()
project = workspace.project("cv-lab-kpdek")
model = project.version(1).model
image_path = r"C:\Users\nagur\Downloads\cat.jpeg"
prediction = model.predict(image_path, confidence=40, overlap=30)
prediction.save("prediction.jpg")
img = Image.open("prediction.jpg")
img.show()

#landing ai
from PIL import Image
from landingai.predict import Predictor
# Enter your API Key
endpoint_id = "f61807dd-dc19-4a71-9448-8890ca451432"
api_key = "land_sk_B2CYuphbrj0fKdA9hHAv6GY4RwRmVnYPBABO8hmrszc3FAFVI6"
# Load your image
image = Image.open(r"C:\Users\nagur\Downloads\cat.jpeg")
# Run inference
predictor = Predictor(endpoint_id, api_key=api_key)
predictions = predictor.predict(image)
print(predictions)
