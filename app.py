# from flask import Flask, render_template, request, redirect, url_for, flash
# import os

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# # Function to check if the file extension is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Route to display the HTML form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle the image upload
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     print("Upload")
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']
#     print(file)

#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         filename = file.filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         return f"File uploaded successfully to aaaaaa{filepath}"
    
#     return 'File type not allowed'

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = r'C:\Users\pc\Desktop\Mtproject\data\project\model\best_model.keras'  # Change this to the correct path
model = load_model(model_path,compile=False)

# Class labels
class_labels = ['10mm', '20mm', '40mm', '60mm']  # Update with your actual class labels

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    img_file = request.files['file']
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load the image
    img_path = os.path.join('uploads', img_file.filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
