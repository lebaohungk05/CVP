import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os
import platform
import psutil
import tensorflow as tf
import json
from datetime import datetime

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def load_dataset(data_dir, target_size=(48, 48), mode='train'):
    """
    Load and preprocess the dataset from directories.
    Args:
        data_dir: Base directory containing train and test folders
        target_size: Size to resize images to
        mode: 'train' or 'test'
    """
    X = []
    y = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    error_count = 0
    print(f"Loading {mode} images from directories...")
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, mode, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory {emotion_dir} does not exist")
            continue
        print(f"Loading {emotion} images...")
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = load_and_preprocess_image(img_path, target_size)
            if img is not None:
                X.append(img)
                y.append(emotion_idx)
            else:
                error_count += 1
    if len(X) == 0:
        raise ValueError(f"No {mode} images were loaded. Please check your data directory structure.")
    print(f"Loaded {len(X)} {mode} images. {error_count} images could not be read.")
    X = np.array(X)
    y = to_categorical(y, num_classes=len(emotions))
    return X, y

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def draw_emotion(frame, face_rect, emotion, confidence):
    """Draw emotion label and confidence on frame."""
    x, y, w, h = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{emotion}: {confidence:.2f}"
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def get_emotion_color(emotion):
    """Get color for different emotions."""
    colors = {
        'angry': (0, 0, 255),    # Red
        'disgust': (0, 128, 0),  # Green
        'fear': (128, 0, 128),   # Purple
        'happy': (0, 255, 255),  # Yellow
        'sad': (255, 0, 0),      # Blue
        'surprise': (255, 165, 0),# Orange
        'neutral': (255, 255, 255)# White
    }
    return colors.get(emotion, (255, 255, 255))

def get_system_info():
    """
    Collect system information for the report
    """
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os": {
            "system": platform.system(),
            "version": platform.version(),
            "machine": platform.machine()
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
            "processor": platform.processor()
        },
        "memory": {
            "total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
        },
        "gpu": {
            "available": len(tf.config.list_physical_devices('GPU')) > 0,
            "devices": [device.name for device in tf.config.list_physical_devices('GPU')]
        },
        "python": {
            "version": platform.python_version()
        },
        "tensorflow": {
            "version": tf.__version__
        }
    }
    
    # Save to file
    os.makedirs("results", exist_ok=True)
    with open("results/system_info.json", "w") as f:
        json.dump(info, f, indent=4)
    
    return info

def print_system_info():
    """
    Print system information in a readable format
    """
    info = get_system_info()
    print("\n=== System Information ===")
    print(f"OS: {info['os']['system']} {info['os']['version']}")
    print(f"CPU: {info['cpu']['processor']}")
    print(f"CPU Cores: {info['cpu']['physical_cores']} physical, {info['cpu']['total_cores']} total")
    print(f"Memory: {info['memory']['total']} total, {info['memory']['available']} available")
    print(f"GPU: {'Available' if info['gpu']['available'] else 'Not available'}")
    if info['gpu']['available']:
        print(f"GPU Devices: {', '.join(info['gpu']['devices'])}")
    print(f"Python Version: {info['python']['version']}")
    print(f"TensorFlow Version: {info['tensorflow']['version']}")
    print("========================\n") 