# Emotion Recognition using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Đồ án môn học: EEE703068-1-3-24(N03) - Thị giác máy tính**  
**Giảng viên hướng dẫn:** Nguyễn Văn Tới  
**Nhóm thực hiện:** Lê Bảo Hưng (Trưởng nhóm), Nguyễn Văn Thái, Nguyễn Quang Hiệp

---

## 📋 Mục lục / Table of Contents
1. [Giới thiệu / Introduction](#giới-thiệu--introduction)
2. [Bài toán / Problem Statement](#bài-toán--problem-statement)
3. [Dữ liệu / Dataset](#dữ-liệu--dataset)
4. [Phương pháp / Methodology](#phương-pháp--methodology)
5. [Kiến trúc Model / Model Architecture](#kiến-trúc-model--model-architecture)
6. [Kết quả / Results](#kết-quả--results)
7. [Cài đặt / Installation](#cài-đặt--installation)
8. [Hướng dẫn sử dụng / Usage](#hướng-dẫn-sử-dụng--usage)
9. [Cấu trúc dự án / Project Structure](#cấu-trúc-dự-án--project-structure)
10. [Đánh giá / Evaluation](#đánh-giá--evaluation)
11. [Kết luận / Conclusions](#kết-luận--conclusions)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

---

## 📖 Giới thiệu / Introduction
Dự án này xây dựng hệ thống nhận diện cảm xúc khuôn mặt sử dụng deep learning (CNN) trên tập dữ liệu FER2013. Hệ thống tự động huấn luyện, đánh giá, lưu lại toàn bộ số liệu, biểu đồ và phân tích chi tiết giúp bạn dễ dàng viết báo cáo khoa học hoặc ứng dụng thực tế.

This project implements a facial emotion recognition system using deep learning (CNN) on the FER2013 dataset. The system automates training, evaluation, and reporting for easy scientific analysis or practical deployment.

---

## 📝 Bài toán / Problem Statement
- **Input**: Ảnh khuôn mặt grayscale 48×48 pixels
- **Output**: Nhãn cảm xúc dự đoán (1 trong 7 cảm xúc: angry, disgust, fear, happy, sad, surprise, neutral)
- **Mục tiêu / Goal**: Xây dựng hệ thống phân loại cảm xúc tự động, tối ưu hóa độ chính xác và hiệu quả tính toán.

---

## 📊 Dữ liệu / Dataset
- **Nguồn / Source**: FER2013 (Facial Expression Recognition 2013)
- **Định dạng / Format**: Ảnh grayscale 48×48 pixels
- **Số lớp / Classes**: 7 cảm xúc cơ bản
- **Cấu trúc thư mục / Directory structure**:
  ```
  data/
    train/
      angry/ disgust/ fear/ happy/ sad/ surprise/ neutral/
    test/
      angry/ ...
  ```
- **Tiền xử lý / Preprocessing**:
  - Đọc ảnh, chuyển grayscale / Read image, convert to grayscale
  - Chuẩn hóa về [0,1] / Normalize to [0,1]
  - One-hot encoding nhãn / One-hot encode labels

---

## 🔬 Phương pháp / Methodology
- **Kiến trúc sử dụng / Architecture**: Convolutional Neural Network (CNN) custom
- **Kỹ thuật / Techniques**: Regularization (Dropout, BatchNorm), EarlyStopping, ReduceLROnPlateau
- **Tối ưu hóa / Optimization**: Adam optimizer, learning rate schedule
- **Đánh giá / Evaluation**: Confusion matrix, classification report, training/validation curves

---

## 🤖 Kiến trúc Model / Model Architecture
```
Input (48×48×1)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(7, softmax)
```
- **Regularization**: Dropout, BatchNorm
- **Optimizer**: Adam (lr=0.0001)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 📈 Kết quả / Results
- **Test Accuracy**: ~46% (FER2013, CNN custom)
- **Best Validation Accuracy**: ~46%
- **Confusion Matrix**: Lưu tại `results/confusion_matrix.png`
- **Chi tiết từng lớp**: Xem `results/classification_report.json`, `results/detailed_metrics.json`

### Key Findings
- Model nhận diện tốt các cảm xúc phổ biến (Happy, Surprise, Neutral)
- Các cảm xúc khó (Disgust, Fear, Sad) thường bị nhầm lẫn
- Dữ liệu mất cân bằng ảnh hưởng lớn đến kết quả

---

## 🚀 Cài đặt / Installation
### Yêu cầu / Requirements
- Python 3.8+
- CUDA GPU (khuyến nghị / recommended)
- RAM >= 8GB

### Hướng dẫn / Instructions
1. **Clone repo**
```bash
git clone https://github.com/lebaohungk05/CV-project.git
cd CV-project
```
2. **Tạo môi trường ảo / Create virtual environment**
```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```
3. **Cài dependencies / Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Chuẩn bị dữ liệu / Prepare data**
- Đặt dữ liệu vào thư mục `data/` theo cấu trúc trên / Place data in `data/` as above

---

## 💻 Hướng dẫn sử dụng / Usage
### Huấn luyện mô hình / Train model
```bash
python train.py
```
- Model sẽ lưu tại `models/emotion_model.h5`
- Kết quả, biểu đồ, báo cáo lưu tại `results/`

### Nhận diện cảm xúc trực tiếp bằng webcam / Real-time emotion detection via webcam
```bash
python detect.py
```
- Nhấn `q` để thoát / Press `q` to quit
- Nhấn `r` để reset bộ đếm cảm xúc / Press `r` to reset emotion counters

### Phân tích kết quả / Analyze results
- Xem các file trong `results/`:
  - `confusion_matrix.png`, `classification_report.json`, `training_plots.png`, ...

---

## 📁 Cấu trúc dự án / Project Structure
```
FER2013-EmotionRecognition/
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── emotion_model.h5
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.json
│   ├── training_plots.png
│   └── ...
├── train.py
├── detect.py
├── model.py
├── utils.py
├── analyze_results.py
├── requirements.txt
└── README.md
```

---

## 📊 Đánh giá / Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Chiến lược / Strategy**: Train/Val/Test split, EarlyStopping, Learning rate schedule
- **Visualization**: Training/validation curves, confusion matrix heatmap

---

## 🎓 Kết luận / Conclusions
- CNN custom đạt kết quả tốt trên FER2013 (~46%)
- Dữ liệu mất cân bằng là thách thức lớn
- Có thể cải thiện thêm bằng transfer learning, augmentation, hoặc tăng dữ liệu cho các lớp khó

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments
- Thanks to Prof. Nguyễn Văn Tới for guidance
- FER2013 dataset creators
- Open-source community for libraries and tools

---

<div align="center">
  <i>For questions or contributions, please open an issue or submit a pull request.</i>
</div>

## ⚠️ Lưu ý khi quản lý mã nguồn với Git / Note on Git source management
- **KHÔNG push file dữ liệu lớn, file nén (archive.zip, *.zip, *.rar, ...) lên GitHub.**
- Nếu lỡ add nhầm, hãy xóa khỏi repo bằng / If you accidentally add, remove from repo with:
  ```bash
  git rm -r --cached archive.zip
  git rm -r --cached '*.zip'
  git commit -m "Remove archive files from repo"
  git push
  ```
- Đảm bảo file `.gitignore` đã có các dòng loại bỏ file nén, dữ liệu, cache.

# Emotion Detection Web Application

A web application for real-time emotion detection using computer vision and deep learning.

## Features

- Real-time emotion detection using webcam
- Upload images for emotion analysis
- Beautiful and responsive UI
- System information display
- Support for 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Webcam Mode**:
   - Click the "Capture" button to take a photo
   - The application will analyze the emotion in real-time

2. **Upload Mode**:
   - Click "Upload Image" to select an image from your computer
   - The application will analyze the emotion in the uploaded image

3. **Results**:
   - The predicted emotion will be displayed
   - A bar chart shows the probability distribution across all emotions
   - System information is displayed at the bottom of the page

## Requirements

- Python 3.7+
- Webcam (for real-time detection)
- Modern web browser with JavaScript enabled

## Note

Make sure to load your trained model in `app.py` by uncommenting and updating the model loading line:
```python
model = tf.keras.models.load_model('path_to_your_model')
```

## Project Structure

```
emotion-detection/
├── app.py              # Main Flask application
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── templates/          # HTML templates
│   └── index.html     # Main page template
└── static/            # Static files (CSS, JS, images)
    └── uploads/       # Uploaded images storage
```

## Contributing

Feel free to submit issues and enhancement requests!