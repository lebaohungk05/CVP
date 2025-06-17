import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
import base64
from utils import load_and_preprocess_image, get_system_info
import tensorflow as tf
from database import init_db, add_user, verify_user, log_emotion, get_user_emotion_history
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Load your trained model here
model = tf.keras.models.load_model('models/emotion_model_cnn.h5')

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize database
init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_id = verify_user(username, password)
        if user_id:
            session['user_id'] = user_id
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if add_user(username, password, email):
            return redirect(url_for('login'))
        return render_template('register.html', error='Username or email already exists')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    history = get_user_emotion_history(user_id)
    return render_template('dashboard.html', history=history)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_image():
    try:
        # Get image data from request
        image_data = request.json['image']
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
            
        # Process the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Extract and preprocess face region
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=[0, -1])
        
        # Get prediction
        prediction = model.predict(face_img, verbose=0)
        
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        predicted_idx = np.argmax(prediction[0])
        predicted_emotion = emotions[predicted_idx]
        confidence = float(prediction[0][predicted_idx])
        
        # Log the emotion
        log_emotion(session['user_id'], predicted_emotion, confidence)
        
        result = {
            'emotions': emotions,
            'probabilities': prediction[0].tolist(),
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'face_coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        }
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/system-info')
@login_required
def system_info():
    return jsonify(get_system_info())

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 