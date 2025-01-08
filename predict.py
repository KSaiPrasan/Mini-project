from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
model = load_model('models/final_model.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model, img_path):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    return prediction

# Path to the new image
img_path =input("Enter the Image path : ")
prediction = predict_image(model, img_path)
print(f'Prediction: {prediction}')
if prediction[0][0] > 0.5:
    print("Prediction: Negative (Clear Skin)")
else:
    print("Prediction: Positive (Cancer Detected)")