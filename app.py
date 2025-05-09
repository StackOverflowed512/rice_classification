from flask import Flask, render_template, request, url_for, flash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'rice_classifier.h5'

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_rice_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
            
        img_class = cv2.resize(img, (64, 64))  # Changed from 28 to 64
        img_class = img_class / 255.0
        img_class = np.expand_dims(img_class, axis=0)
        
        prediction = model.predict(img_class)
        rice_types = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        predicted_type = rice_types[np.argmax(prediction)]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        whole_count = 0
        broken_count = 0
        
        areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 100]
        avg_area = np.mean(areas) if areas else 0
        area_threshold = avg_area * 0.7
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            is_broken = area < area_threshold
            
            x, y, w, h = cv2.boundingRect(contour)
            color = (0, 0, 255) if is_broken else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            if is_broken:
                broken_count += 1
            else:
                whole_count += 1
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.jpg')
        cv2.imwrite(output_path, img)
        
        return {
            'rice_type': predicted_type,
            'whole_count': whole_count,
            'broken_count': broken_count,
            'processed_image': 'processed.jpg',
            'confidence': float(np.max(prediction))
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        flash('Model not loaded. Please check server logs.', 'error')
        return render_template('index.html')
        
    if 'image' not in request.files:
        flash('No image uploaded', 'error')
        return render_template('index.html')
        
    file = request.files['image']
    if file.filename == '':
        flash('No selected file', 'error')
        return render_template('index.html')
        
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload JPG or PNG images.', 'error')
        return render_template('index.html')
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = analyze_rice_image(filepath)
        return render_template('results.html', results=results)
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        flash('Error processing image. Please try again.', 'error')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)