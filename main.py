import numpy as np
import tensorflow as tf
from PIL import Image
import sys

MODEL_PATH = './model/model_emosi.keras'  
IMAGE_PATH = './download.jpeg'        
INPUT_SIZE = (48, 48)          
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  
    image = image.resize(INPUT_SIZE)
    img_array = np.array(image) / 255.0          
    img_array = img_array.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 1) 
    return img_array

def predict(image_path):
    input_image = preprocess_image(image_path)
    prediction = model.predict(input_image)[0]
    predicted_label = LABELS[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    print(f"Ekspresi: {predicted_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Vector prediksi: {prediction}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    predict(IMAGE_PATH)
