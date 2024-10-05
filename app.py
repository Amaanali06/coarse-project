from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os


# from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

model_path = r'/Users/mohammedyasinmulla/Documents/coarse-project/model/best_model.keras'
model = load_model(model_path, compile=False)
class_labels = ['class1', 'class2','class3', 'class4']  # Replace with your actual class labels


# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to display the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def predict_file():
    print("Upload")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    print(file)

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return f"File uploaded successfully to aaaaaa{filepath}"
    
    return 'File type not allowed'



@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        print("Upload")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load the image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            print("predicted_class", predicted_class)
            return jsonify({'predicted_class': predicted_class})

        return 'File type not allowed'
    except Exception as e:
        print(f"An error occurred during image processing or prediction: {e}")
        return 'An error occurred during image processing or prediction'
# @app.route('/predict', methods=['POST'])
# def predict():
#     print("test")
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']
#     print("Upload")

#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         filename = file.filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Load the image
#         img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img) / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Make prediction
#         predictions = model.predict(img_array)
#         predicted_class = class_labels[np.argmax(predictions)]
#         print("predicted_class", predicted_class)
#         return jsonify({'predicted_class': predicted_class})

#     return 'File type not allowed'


if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

# # from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# app = Flask(__name__)

# # Load the model
# # model_path = r'C:\Users\pc\Desktop\Mtproject\data\project\model\best_model.keras'  # Change this to the correct path
# model_path = r'/Users/mohammedyasinmulla/Documents/coarse-project/model/best_model.keras'
# model = load_model(model_path,compile=False)

# # Class labels
# class_labels = ['10mm', '20mm', '40mm', '60mm']  # Update with your actual class labels
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     img_file = request.files['file']
#     if img_file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Load the image
#     img_path = os.path.join('uploads', img_file.filename)
#     img_file.save(img_path)

#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make prediction
#     predictions = model.predict(img_array)
#     predicted_class = class_labels[np.argmax(predictions)]

#     return jsonify({'predicted_class': predicted_class})

# if __name__ == '__main__':
#     app.run(debug=True)
