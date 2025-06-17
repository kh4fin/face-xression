from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io

app = Flask(__name__)
model = tf.keras.models.load_model('./model/model_emosi.keras')  

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1] 
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L') 
    image = image.resize((48, 48))  

    img_array = np.array(image) / 255.0
    img_array = image.reshape(1, 48, 48, 1)

    prediction = model.predict(img_array)[0]
    label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
