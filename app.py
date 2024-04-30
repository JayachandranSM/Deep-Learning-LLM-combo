from flask import Flask, render_template, request, redirect, url_for
from collections import Counter
from ultralytics import YOLO
import openai
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
from flask import jsonify
from flask import render_template




# Create Flask application instance
app = Flask(__name__, static_folder='static')

# Define a custom filter to calculate the length
@app.template_filter('length')
def length(obj):
    return len(obj)

# Set up OpenAI API key  
openai.api_key = "*****"



# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def boundingboxPredicted(results, model, image_path):
    output_folder = 'static\predictions'  # Define the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)

    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

        img = annotator.result()

    # Save the original image with bounding boxes
    output_image_path = os.path.join(output_folder, 'predictions.jpg')
    cv2.imwrite(output_image_path, img)
    print(f"Predictions saved in {output_folder}")
    return output_image_path
    
# Function to run object detection
def run_object_detection(image_path):
    model_directory = r"E:\Pain_detection_yolov8\YOLOv8-Image-Segmentation\runs\detect\train\weights"
    model_filename = "best.pt"
    model_path = os.path.join(model_directory, model_filename)

    infer = YOLO(model_path)
    result = infer.predict(image_path)
    # Get object counts
    item_counts = Counter(infer.names[int(c)] for r in result for c in r.boxes.cls)
    object_list = list(item_counts.keys())
    return result, object_list, infer

# Function to handle home page request
@app.route('/')
def index():
    # Render home page template
    return render_template('index.html')

# Function to handle image processing request
@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return render_template('result.html', error="No file part")

    image_file = request.files['image']

    # If the user does not select a file, browser also
    # submit an empty part without filename
    if image_file.filename == '':
        return render_template('result.html', error="No selected file")

    # Check if the file is allowed (optional)
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template('result.html', error="Invalid file type")

    # Save the image
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the image with a fixed filename
    image_path = os.path.join(upload_dir, 'uploaded_image.jpg')
    image_file.save(image_path)

    # Run object detection on the uploaded image
    result, object_list, infer = run_object_detection(image_path)

    # Construct prompt based on detection results
    prompt_pain = "To alleviate pain, try deep breathing and mindfulness techniques, coupled with gentle stretching exercises to relax tense muscles. Remember to stay hydrated and get adequate rest to promote healing and reduce discomfort."
    prompt_no_pain = "Practice regular exercise and maintain a balanced diet for overall health and well-being. Stay up to date on health screenings and vaccinations to prevent illness and promote early detection."

    # Get chatbot responses using OpenAI API
    # Construct responses
    responses = []
    if 'Pain' in object_list:
        responses.append({
            'title': 'Pain',
            'class': 'pain-response',
            'content': prompt_pain
        })
    
    if 'No-pain' in object_list:
        responses.append({
            'title': 'No-pain',
            'class': 'no-pain-response',
            'content': prompt_no_pain
        })


    # Get the predicted image path
    predicted_image_path = boundingboxPredicted(result, infer, image_path)

    # Pass variables to the result template
    return render_template('result.html', responses=responses, image_filename=image_file.filename, object_list=object_list, predicted_image_path=predicted_image_path)







