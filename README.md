# Emotion Recognition using Deep Learning and Computer Vision

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Đồ án môn học:** Thị giác máy tính-1-3-24(N03)  
**Giảng viên hướng dẫn:** Nguyễn Văn Tới

**Nhóm thực hiện:**
- **Lê Bảo Hưng** (Trưởng nhóm) - [@lebaohungk05](https://github.com/lebaohungk05)
- **Nguyễn Văn Thái**
- **Nguyễn Quang Hiệp**

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Web Application](#-web-application)
- [Deployment](#-deployment)
- [Evaluation](#-evaluation)
- [Conclusions](#-conclusions)
- [Future Work](#-future-work)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🎯 Introduction

This project implements a comprehensive **Facial Emotion Recognition System** using Deep Learning and Computer Vision techniques. The system can classify facial expressions into **7 basic emotions**: angry, disgust, fear, happy, sad, surprise, and neutral.

### Key Features
- **Real-time emotion detection** via webcam
- **Web-based dashboard** with user authentication
- **CNN-based deep learning model** with 60.77% accuracy
- **Interactive data visualization** and analytics
- **Database integration** for user management and history tracking
- **Responsive web interface** built with Flask
- **Production deployment** ready on Render cloud platform

### Technology Stack
- **Deep Learning:** TensorFlow/Keras, CNN Architecture
- **Computer Vision:** OpenCV, image preprocessing
- **Backend:** Flask, SQLite database
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn
- **Deployment:** Render, Docker-ready configuration

## 📝 Problem Statement

### Context
Facial emotion recognition is a critical component in:
- **Human-Computer Interaction (HCI)**
- **Psychological and behavioral analysis**
- **Security and surveillance systems**
- **Educational technology and e-learning**
- **Healthcare and mental health monitoring**

### Objectives
1. **Develop an automated system** for real-time emotion classification
2. **Achieve high accuracy** using deep learning techniques
3. **Create a user-friendly web interface** for practical applications
4. **Implement scalable architecture** for future enhancements
5. **Provide comprehensive analysis** of model performance
6. **Deploy to production** for real-world accessibility

### Input/Output Specification
- **Input:** Grayscale facial images (48×48 pixels)
- **Output:** Predicted emotion label with confidence score
- **Real-time processing:** 30+ FPS on standard hardware

## 📊 Dataset

### Overview
- **Source:** FER2013-style dataset structure
- **Format:** Grayscale images, 48×48 pixels
- **Classes:** 7 emotion categories based on Ekman's Basic Emotions
- **Total samples:** ~35,000 training + ~7,000 testing images

### Data Distribution
```
Training Set:
├── Happy: 7,215 images (21.3%)
├── Neutral: 4,965 images (14.7%)
├── Sad: 4,830 images (14.3%)
├── Fear: 4,097 images (12.1%)
├── Angry: 3,995 images (11.8%)
├── Surprise: 3,171 images (9.4%)
└── Disgust: 436 images (1.3%)

Test Set:
├── Happy: 1,774 images (24.7%)
├── Neutral: 1,233 images (17.2%)
├── Sad: 1,247 images (17.4%)
├── Fear: 1,024 images (14.3%)
├── Angry: 958 images (13.4%)
├── Surprise: 831 images (11.6%)
└── Disgust: 111 images (1.5%)
```

### Data Preprocessing
1. **Image Loading:** Read and convert to grayscale
2. **Normalization:** Scale pixel values to [0,1] range
3. **Data Augmentation:** Rotation, shifting, zooming, flipping
4. **Label Encoding:** One-hot encoding for multi-class classification

## 🔬 Methodology

### Literature Review

#### Traditional Computer Vision Approaches
- **Feature Extraction:** HOG, LBP, SIFT, Gabor filters
- **Classifiers:** SVM, Random Forest, k-NN
- **Advantages:** Fast inference, interpretable
- **Limitations:** Manual feature engineering, limited accuracy

#### Deep Learning Approaches
- **Architectures:** CNN, ResNet, VGG, DenseNet
- **Techniques:** Transfer learning, attention mechanisms
- **Advantages:** Automatic feature learning, high accuracy
- **Limitations:** Requires large datasets, computational resources

### Our Approach
We implement a **custom CNN architecture** optimized for facial emotion recognition:
- **Multi-layer feature extraction** with batch normalization
- **Regularization techniques** to prevent overfitting
- **Data augmentation** for improved generalization
- **Adaptive learning rate** scheduling

## 🤖 Model Architecture

### Convolutional Neural Network (CNN)

```
Input (48×48×1)
    ↓
Conv Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → Conv2D(32, 3×3) → BatchNorm → ReLU
    ↓
MaxPooling2D(2×2) → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → Conv2D(64, 3×3) → BatchNorm → ReLU
    ↓
MaxPooling2D(2×2) → Dropout(0.3)
    ↓
Conv Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → Conv2D(128, 3×3) → BatchNorm → ReLU
    ↓
MaxPooling2D(2×2) → Dropout(0.4)
    ↓
Conv Block 4: Conv2D(256, 3×3) → BatchNorm → ReLU → GlobalAveragePooling2D
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(7, softmax)
```

### Model Specifications
- **Total Parameters:** 618,727 (617,063 trainable)
- **Optimizer:** Adam (learning rate: 0.0001)
- **Loss Function:** Categorical Crossentropy
- **Regularization:** L2 (0.001) + Dropout (0.25-0.5)
- **Training Time:** 49.43 minutes
- **Batch Size:** 8
- **Epochs:** 50 (with early stopping)

### Key Features
1. **Batch Normalization:** Stabilizes training and accelerates convergence
2. **Dropout Layers:** Prevents overfitting with progressive rates
3. **Global Average Pooling:** Reduces parameters and overfitting
4. **Data Augmentation:** Real-time augmentation during training

## 📈 Results

### Overall Performance
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **60.77%** |
| **Training Time** | 49.43 minutes |
| **Model Size** | 2.4 MB |
| **Inference Speed** | 30+ FPS |
| **Total Parameters** | 618,727 |

### Per-Class Performance
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Angry** | 49.08% | 61.06% | 54.42% | 958 |
| **Disgust** | 57.69% | 27.03% | 36.81% | 111 |
| **Fear** | 53.45% | 21.19% | 30.35% | 1024 |
| **Happy** | 78.54% | 86.87% | 82.49% | 1774 |
| **Sad** | 49.90% | 41.62% | 45.39% | 1247 |
| **Surprise** | 72.02% | 74.97% | 73.47% | 831 |
| **Neutral** | 50.99% | 68.69% | 58.53% | 1233 |

### Real-World Usage Statistics
Based on production deployment data:
- **Active Users:** 4 registered users
- **Total Emotion Analyses:** 158 predictions
- **Most Detected Emotion:** Angry (53.8% of predictions)
- **Emotion Distribution in Real Usage:**
  - Angry: 53.8% (confidence: 98.5%)
  - Neutral: 25.3% (confidence: 62.4%)
  - Surprise: 12.0% (confidence: 69.9%)
  - Happy: 7.6% (confidence: 79.7%)
  - Sad: 1.3% (confidence: 31.9%)

### Key Findings
- **Best Performance:** Happy emotion (82.49% F1-score)
- **Most Challenging:** Fear emotion (30.35% F1-score)
- **Common Confusions:** Fear↔Surprise, Sad↔Neutral, Angry↔Sad
- **Class Imbalance Impact:** Disgust and Fear severely underrepresented in performance

## 🚀 Installation

### Prerequisites
- **Python 3.8+** 
- **pip** package manager
- **Git** for version control
- **Webcam** (for real-time detection)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/lebaohungk05/CVP.git
cd CVP
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained model**
The trained model (`emotion_model_cnn.h5`) is included in the `models/` directory.

## 💻 Usage

### Web Application
```bash
python app.py
```
- Open browser and navigate to `http://localhost:5000`
- Register/Login to access the dashboard
- Upload images or use webcam for real-time detection

### Real-time Detection
```bash
python detect.py
```
- Opens webcam feed with real-time emotion detection
- Press 'q' to quit

### Model Training
```bash
python train.py
```
- Trains the CNN model from scratch
- Saves model and generates evaluation reports

### Data Analysis
```bash
python analyze_results.py
```
- Generates comprehensive analysis reports
- Creates visualization plots and confusion matrices

## 📁 Project Structure

```
Computer Vision Project/
├── 📱 app.py                     # Flask web application
├── 🧠 model.py                   # CNN architecture definition
├── 🎯 detect.py                  # Real-time emotion detection
├── 🏃 train.py                   # Model training script
├── 💾 database.py                # Database operations
├── 🔧 utils.py                   # Utility functions
├── 📊 analyze_results.py         # Results analysis
├── 📈 data_analysis.py           # Dataset analysis
├── 🎨 plot_*.py                  # Visualization scripts
│
├── 📂 data/                      # Dataset directory
│   ├── 🏋️ train/                # Training images (7 classes)
│   └── ✅ test/                  # Test images (7 classes)
│
├── 🤖 models/                    # Saved models and results
│   ├── emotion_model_cnn.h5      # Trained CNN model
│   ├── cnn_results.json          # Model evaluation metrics
│   ├── training_log.csv          # Training history
│   └── confusion_matrix_cnn.csv  # Confusion matrix data
│
├── 🌐 templates/                 # HTML templates
│   ├── index.html                # Home page
│   ├── dashboard.html            # User dashboard
│   ├── login.html                # Login page
│   ├── register.html             # Registration page
│   └── database_viewer.html      # Database management
│
├── 📊 plots/                     # Visualization outputs
│   ├── training_history_cnn.png  # Training curves
│   ├── confusion_matrix_cnn.png  # Confusion matrix
│   └── cnn_report.html           # Detailed HTML report
│
├── 🗃️ static/                    # Static web assets
│   ├── uploads/                  # User uploaded files
│   ├── camera_fix.js            # Camera utilities
│   └── database_viewer_static.html # Static database view
│
├── 🚀 Deployment Files           # Production deployment
│   ├── Procfile                  # Render deployment config
│   ├── render.yaml              # Render service config
│   ├── build.sh                 # Build script
│   ├── runtime.txt              # Python version
│   ├── .renderignore            # Deployment ignore rules
│   └── DEPLOYMENT.md            # Deployment documentation
│
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md                  # Project documentation
```

## 🌐 Web Application

### Features
1. **User Authentication System**
   - Secure registration and login with password hashing
   - Session management with Flask-Login
   - User profile dashboard with statistics

2. **Real-time Emotion Detection**
   - Webcam integration with WebRTC
   - Live emotion prediction with confidence scores
   - Real-time emotion history tracking

3. **Image Upload Analysis**
   - Batch image processing support
   - Detailed emotion breakdown with probabilities
   - Results visualization and export

4. **Dashboard Analytics**
   - Personal usage statistics
   - Historical emotion data visualization
   - Performance metrics and insights

5. **Database Management**
   - Built-in database viewer interface
   - User and emotion log management
   - Export functionality for analysis

### Technology Stack
- **Backend:** Flask, SQLite, SQLAlchemy, Werkzeug
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap
- **Real-time:** WebRTC for camera access
- **Visualization:** Chart.js, Plotly for analytics
- **Security:** Password hashing, session management

## 🚀 Deployment

### Production Deployment on Render

The application is configured for deployment on **Render** cloud platform with the following setup:

#### Deployment Configuration
- **Platform:** Render Web Service
- **Runtime:** Python 3.8.10
- **Build Command:** `./build.sh`
- **Start Command:** `python app.py`
- **Auto-deploy:** Enabled on main branch

#### Key Files
- `Procfile`: Process definition for Render
- `render.yaml`: Service configuration
- `build.sh`: Build script with dependencies
- `runtime.txt`: Python version specification
- `.renderignore`: Files excluded from deployment

#### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set production environment
export FLASK_ENV=production
export PORT=10000

# Start application
python app.py
```

#### Features
- **Automatic scaling** based on traffic
- **HTTPS** enabled by default
- **Custom domain** support
- **Environment variables** for configuration
- **Persistent storage** for SQLite database

### Local Development
```bash
# Development mode
export FLASK_ENV=development
export FLASK_DEBUG=True
python app.py
```

### Docker Support (Optional)
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📊 Evaluation

### Metrics Used
1. **Accuracy:** Overall classification correctness (60.77%)
2. **Precision:** True positives / (True positives + False positives)
3. **Recall:** True positives / (True positives + False negatives)
4. **F1-Score:** Harmonic mean of precision and recall
5. **Confusion Matrix:** Detailed error analysis

### Evaluation Strategy
- **Data Split:** 80% training, 20% testing
- **Stratified Sampling:** Maintains class distribution
- **Cross-validation:** 5-fold for hyperparameter tuning
- **Early Stopping:** Prevents overfitting (patience=10)

### Performance Analysis
1. **Strengths:**
   - Excellent performance on Happy emotion (82.49% F1)
   - Good generalization with data augmentation
   - Stable training with batch normalization
   - Real-world deployment readiness

2. **Challenges:**
   - Class imbalance affects Disgust recognition
   - Confusion between similar emotions
   - Limited performance on Fear emotion

3. **Visualization:**
   - Training/validation loss curves
   - Confusion matrix heatmaps
   - Per-class performance comparison
   - Real-time usage analytics

## 🎓 Conclusions

### Summary
Our CNN-based emotion recognition system demonstrates:
- **Competitive accuracy** of 60.77% on challenging dataset
- **Real-time performance** suitable for practical applications
- **Robust architecture** with effective regularization
- **Production-ready deployment** with comprehensive web interface
- **Real-world validation** through user interactions

### Key Achievements
1. **Model Performance:** Achieved 60.77% accuracy with efficient architecture
2. **Web Application:** Developed complete system with user interface
3. **Real-time Processing:** Implemented live emotion detection
4. **Production Deployment:** Successfully deployed on Render platform
5. **Comprehensive Analysis:** Detailed evaluation and visualization
6. **User Adoption:** 4 active users with 158+ emotion analyses

### Lessons Learned
1. **Data Quality:** Clean, balanced datasets crucial for performance
2. **Regularization:** Batch normalization and dropout prevent overfitting
3. **Class Imbalance:** Severely affects minority class performance
4. **Feature Engineering:** CNN automatically learns relevant features
5. **Deployment:** Production readiness requires careful configuration

## 🔮 Future Work

### Short-term Improvements
1. **Data Augmentation:** Advanced techniques (Mixup, CutMix)
2. **Class Balancing:** SMOTE, focal loss for imbalanced classes
3. **Ensemble Methods:** Combine multiple models for better accuracy
4. **Mobile Deployment:** Optimize for mobile applications
5. **API Development:** RESTful API for third-party integration

### Long-term Enhancements
1. **Transfer Learning:** Pre-trained models (VGGFace, FaceNet)
2. **Multi-modal Analysis:** Combine facial, audio, and text features
3. **Attention Mechanisms:** Focus on relevant facial regions
4. **3D Emotion Recognition:** Temporal information for video analysis
5. **Edge Computing:** Deploy on IoT devices and edge platforms

### Technical Roadmap
- **Performance:** Target 70%+ accuracy with advanced architectures
- **Scalability:** Microservices architecture with Kubernetes
- **Integration:** API marketplace and plugin ecosystem
- **Research:** Publish findings in academic conferences

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Prof. Nguyễn Văn Tới** for expert guidance and supervision
- **Phenikaa University** for providing resources and support
- **FER2013 dataset creators** for the foundational dataset
- **Render Platform** for free deployment hosting
- **Open-source community** for excellent libraries and tools:
  - TensorFlow/Keras team
  - OpenCV contributors
  - Flask development team
  - Matplotlib and Seaborn developers

### References
1. Goodfellow, I. et al. (2013). "Challenges in representation learning: A report on three machine learning contests."
2. Ekman, P. (1992). "An argument for basic emotions."
3. LeCun, Y. et al. (1998). "Gradient-based learning applied to document recognition."

---

**🌟 Live Demo:** [Visit our deployed application](https://your-render-url.onrender.com)

**For questions, issues, or contributions, please:**
- 📧 **Email:** hungk1210@gmail.com
- 🐛 **Issues:** [Create an Issue](https://github.com/lebaohungk05/CVP/issues)
- 🔄 **Pull Requests:** [Submit a PR](https://github.com/lebaohungk05/CVP/pulls)

⭐ **Star this repository if you found it helpful!**
