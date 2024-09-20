from flask import Flask, request, jsonify , render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model
model = load_model('model/cnn-cataract-eye-disease.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = request.files['image']

    # Save the image temporarily
    img_path = os.path.join('temp', img.filename)
    img.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).resize((224, 224))  # Adjust size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image
    img_array = img_array / 255.0

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Clean up the temporary image
    os.remove(img_path)
    prediction = prediction[0]
    # result = {
    #     'normal' : str(prediction[0]),
    #     'cataract' : str(prediction[1]),
    #     'glaucoma' : str(prediction[2]),
    #     'retinaDisease': str(prediction[3]),
    # }
    result = {
        'cataract' : str(prediction[0]),
        'normal': str(prediction[1]),
    }
    print(result)
    # Return the prediction
    return jsonify(result)
if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)
