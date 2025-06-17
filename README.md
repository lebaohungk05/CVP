# Emotion Detection Project

Dự án phát hiện cảm xúc khuôn mặt (Facial Emotion Detection) sử dụng Deep Learning và Computer Vision. Ứng dụng có thể nhận diện 7 cảm xúc cơ bản: Angry (Giận dữ), Disgust (Ghê tởm), Fear (Sợ hãi), Happy (Vui vẻ), Sad (Buồn), Surprise (Ngạc nhiên), và Neutral (Trung tính).

## Tính năng

- Nhận diện cảm xúc thời gian thực qua webcam
- Giao diện web thân thiện với người dùng
- Đăng ký và đăng nhập người dùng
- Dashboard hiển thị thống kê và kết quả phân tích
- Hỗ trợ lưu trữ lịch sử nhận diện

## Thông số kỹ thuật

- **Độ chính xác trên tập test**: 60.43%
- **Thời gian huấn luyện**: 103.28 phút
- **Tổng số tham số**: 617,063
- **Các lớp cảm xúc**:
  - Angry: Precision 50.20%, Recall 52.61%
  - Disgust: Precision 56.41%, Recall 19.82%
  - Fear: Precision 52.76%, Recall 25.20%
  - Happy: Precision 82.87%, Recall 84.84%
  - Sad: Precision 50.80%, Recall 40.82%
  - Surprise: Precision 74.30%, Recall 73.77%
  - Neutral: Precision 46.28%, Recall 75.18%

## Cài đặt và Chạy Locally

1. Clone repository:
```bash
git clone https://github.com/lebaohungk05/CVP.git
cd CVP
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

4. Chạy ứng dụng:
```bash
python app.py
```

## Cấu trúc Project

```
Computer Vision Project/
├── app.py                 # Flask application chính 
├── model.py              # Model architecture và training
├── detect.py            # Logic nhận diện cảm xúc
├── database.py          # Database operations
├── utils.py             # Utility functions
├── data/                # Training và test data
├── models/              # Trained models
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates
└── results/            # Kết quả phân tích và đánh giá
```

## Deployment

Dự án được deploy trên Render.com. Các bước deploy:

1. Đảm bảo repository GitHub được cập nhật với code mới nhất
2. Tạo một Web Service mới trên Render
3. Kết nối với repository GitHub
4. Cấu hình:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Python Version: 3.9.18 (specified in runtime.txt)

## License

Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Tác giả

- Le Bao Hung (GitHub: [@lebaohungk05](https://github.com/lebaohungk05))