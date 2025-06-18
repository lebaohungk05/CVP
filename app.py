import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import cv2
import numpy as np
import base64
from utils import get_emotion_color
import tensorflow as tf
from database import init_db, add_user, verify_user, log_emotion, get_user_emotion_history
from functools import wraps
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load your trained model
try:
    model = tf.keras.models.load_model('models/emotion_model_cnn.h5')
    print("✅ Model loaded successfully!")
except:
    try:
        model = tf.keras.models.load_model('models/emotion_model.h5')
        print("✅ Alternative model loaded successfully!")
    except:
        print("❌ No model found! Please train the model first.")
        model = None

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize database
init_db()

# Emotion labels and colors
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_COLORS = {
    'angry': '#FF6B6B',
    'disgust': '#4ECDC4', 
    'fear': '#45B7D1',
    'happy': '#FFA07A',
    'sad': '#98D8C8',
    'surprise': '#F7DC6F',
    'neutral': '#BB8FCE'
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this feature.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('detect'))
    return redirect(url_for('login'))

@app.route('/detect')
@login_required
def detect():
    """Main emotion detection page - login required"""
    username = session.get('username', 'User')
    return render_template('index.html', 
                         emotions=EMOTIONS, 
                         colors=EMOTION_COLORS,
                         username=username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('detect'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_id = verify_user(username, password)
        if user_id:
            session['user_id'] = user_id
            session['username'] = username
            flash('Login successful! Welcome back!', 'success')
            return redirect(url_for('detect'))
        flash('Invalid username or password. Please try again.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('detect'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
        elif add_user(username, password, email):
            flash('Registration successful! Please login with your credentials.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists. Please try different credentials.', 'error')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    username = session.get('username', 'User')
    session.clear()
    flash(f'Goodbye {username}! You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    username = session.get('username', 'User')
    history = get_user_emotion_history(user_id)
    
    # Calculate emotion statistics
    if history:
        emotion_counts = {}
        for record in history:
            emotion = record[1]  # emotion column
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_detections = len(history)
        emotion_stats = {
            emotion: {
                'count': count,
                'percentage': round((count / total_detections) * 100, 1)
            }
            for emotion, count in emotion_counts.items()
        }
    else:
        emotion_stats = {}
        total_detections = 0
    
    return render_template('dashboard.html', 
                         username=username,
                         history=history, 
                         emotion_stats=emotion_stats,
                         total_detections=total_detections,
                         colors=EMOTION_COLORS)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_image():
    """Analyze emotion from uploaded image - login required"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500
            
        # Get image data from request
        if 'image' in request.json:
            # Base64 image from webcam
            image_data = request.json['image']
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
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
            return jsonify({'error': 'No face detected. Please ensure your face is clearly visible and well-lit.'}), 400
            
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
        predicted_idx = np.argmax(prediction[0])
        predicted_emotion = EMOTIONS[predicted_idx]
        confidence = float(prediction[0][predicted_idx])
        
        # Log the emotion to user's history
        log_emotion(session['user_id'], predicted_emotion, confidence)
        
        # Prepare all emotion probabilities
        emotion_probabilities = [
            {
                'emotion': emotion,
                'probability': float(prediction[0][i]),
                'percentage': round(float(prediction[0][i]) * 100, 1),
                'color': EMOTION_COLORS[emotion]
            }
            for i, emotion in enumerate(EMOTIONS)
        ]
        
        # Sort by probability (highest first)
        emotion_probabilities.sort(key=lambda x: x['probability'], reverse=True)
        
        result = {
            'success': True,
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'confidence_percentage': round(confidence * 100, 1),
            'emotion_probabilities': emotion_probabilities,
            'face_coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'color': EMOTION_COLORS[predicted_emotion],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': session.get('username', 'User')
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/model-info')
@login_required
def model_info():
    """Get information about the loaded model - login required"""
    if model is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    # Load model stats
    try:
        with open('models/classification_report_cnn.txt', 'r') as f:
            report_text = f.read()
    except:
        report_text = "Model report not available"
    
    try:
        with open('results/dataset_info.json', 'r') as f:
            dataset_info = json.load(f)
    except:
        dataset_info = {}
    
    return jsonify({
        'model_loaded': True,
        'emotions': EMOTIONS,
        'total_parameters': 617063,
        'dataset_info': dataset_info,
        'accuracy': '60.77%',
        'report': report_text,
        'user': session.get('username', 'User')
    })

@app.route('/api/user-stats')
@login_required 
def user_stats():
    """Get current user's emotion statistics"""
    user_id = session['user_id']
    history = get_user_emotion_history(user_id)
    
    if not history:
        return jsonify({'total': 0, 'emotions': {}})
    
    emotion_counts = {}
    for record in history:
        emotion = record[1]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return jsonify({
        'total': len(history),
        'emotions': emotion_counts,
        'user': session.get('username', 'User')
    })

if __name__ == '__main__':
    print("🚀 Starting Emotion Recognition Web App...")
    print("🔐 Login required for all features")
    print("📊 Model status:", "✅ Loaded" if model is not None else "❌ Not loaded")
    app.run(debug=True, host='0.0.0.0', port=5000) 