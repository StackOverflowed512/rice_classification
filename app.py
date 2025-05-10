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
RICE_TYPE_MODEL = 'rice_classifier.h5'
CONDITION_MODEL = 'condition_classifier.h5'

try:
    type_model = load_model(RICE_TYPE_MODEL)
    condition_model = load_model(CONDITION_MODEL)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    type_model = None
    condition_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_rice_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
            
        # Rice type classification
        img_type = cv2.resize(img, (64, 64))
        img_type = img_type / 255.0
        img_type = np.expand_dims(img_type, axis=0)
        
        type_prediction = type_model.predict(img_type)
        rice_types = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        predicted_type = rice_types[np.argmax(type_prediction)]
        type_confidence = float(np.max(type_prediction))
        
        # Process image for grain detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize counters
        broken_count = 0
        damaged_count = 0
        discolored_count = 0
        normal_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip small contours
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            grain_img = img[y:y+h, x:x+w]
            
            # Classify grain condition
            grain_resized = cv2.resize(grain_img, (128, 128))
            grain_normalized = grain_resized / 255.0
            grain_normalized = np.expand_dims(grain_normalized, axis=0)
            
            condition_prediction = condition_model.predict(grain_normalized)
            condition_idx = np.argmax(condition_prediction)
            condition_conf = float(np.max(condition_prediction))
            
            # Color coding based on condition
            if condition_idx == 0 and condition_conf > 0.5:  # Broken
                color = (0, 0, 255)  # Red
                broken_count += 1
            elif condition_idx == 1 and condition_conf > 0.5:  # Damaged
                color = (0, 165, 255)  # Orange
                damaged_count += 1
            elif condition_idx == 2 and condition_conf > 0.5:  # Discolored
                color = (0, 255, 255)  # Yellow
                discolored_count += 1
            else:
                color = (0, 255, 0)  # Green for normal
                normal_count += 1
            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.jpg')
        cv2.imwrite(output_path, img)
        
        return {
            'rice_type': predicted_type,
            'type_confidence': type_confidence,
            'normal_count': normal_count,
            'broken_count': broken_count,
            'damaged_count': damaged_count,
            'discolored_count': discolored_count,
            'processed_image': 'processed.jpg'
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if type_model is None or condition_model is None:
        flash('Models not loaded. Please check server logs.', 'error')
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