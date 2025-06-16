import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, BatchNormalization
from model import create_emotion_model, save_model
from utils import load_dataset, plot_training_history, get_system_info, print_system_info
from analyze_results import analyze_training_results
import tensorflow as tf
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Enable memory growth for GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def augment_data(image, label):
    # Không augmentation, trả về ảnh gốc
    return image, label

def create_dataset(X, y, batch_size, augment=False):
    """
    Create a tf.data.Dataset with optional augmentation
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def plot_and_save_class_distribution(y, class_names, title, filename):
    counts = np.sum(y, axis=0)
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{title}: {dict(zip(class_names, counts))}")

def train_model(data_dir, model_save_path, batch_size=64, epochs=100):
    """
    Train the emotion recognition model with optimizations for laptop
    """
    # Print system information
    print_system_info()
    
    # Load and preprocess dataset
    print("Loading training dataset...")
    X_train, y_train = load_dataset(data_dir, mode='train')
    print("Loading test dataset...")
    X_test, y_test = load_dataset(data_dir, mode='test')

    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Calculate class weights
    class_weights = {
        0: 1.0,  # Angry
        1: 7.0,  # Disgust (7178/111 ≈ 7)
        2: 1.0,  # Fear
        3: 0.4,  # Happy (7178/1774 ≈ 0.4)
        4: 1.0,  # Sad
        5: 1.0,  # Surprise
        6: 1.0   # Neutral
    }

    # Stratified split for train/val
    y_train_labels = np.argmax(y_train, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(X_train, y_train_labels):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Plot and print class distributions
    os.makedirs("results", exist_ok=True)
    plot_and_save_class_distribution(y_tr, class_names, "Train Set Class Distribution", "results/train_class_dist.png")
    plot_and_save_class_distribution(y_val, class_names, "Validation Set Class Distribution", "results/val_class_dist.png")
    plot_and_save_class_distribution(y_test, class_names, "Test Set Class Distribution", "results/test_class_dist.png")

    # Reshape data for CNN input
    X_tr = X_tr.reshape(X_tr.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    X_tr = X_tr.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Create datasets
    train_dataset = create_dataset(X_tr, y_tr, batch_size, augment=True)
    val_dataset = create_dataset(X_val, y_val, batch_size, augment=False)
    test_dataset = create_dataset(X_test, y_test, batch_size, augment=False)
    
    # Create model with optimizations
    print("Creating model...")
    model = create_emotion_model()
    
    # Define callbacks with optimized parameters
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model with optimized parameters
    print("Training model...")
    start_time = datetime.now()
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save training results
    training_results = {
        "training_time_seconds": training_time,
        "final_epoch": len(history.history['loss']),
        "early_stopping_epoch": len(history.history['loss']) if history.history['loss'][-1] == min(history.history['loss']) else None,
        "final_learning_rate": float(model.optimizer.learning_rate.numpy()),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "best_val_loss": float(min(history.history['val_loss']))
    }
    
    with open("results/training_results.json", "w") as f:
        json.dump(training_results, f, indent=4)
    
    # Analyze and save results
    print("\nAnalyzing and saving results...")
    final_metrics = analyze_training_results(history, model, X_test, y_test)
    
    # Save final model
    save_model(model, model_save_path)
    
    return model, history, final_metrics

if __name__ == "__main__":
    # Define paths
    DATA_DIR = "data"  # Path to directory containing train and test folders
    MODEL_SAVE_PATH = "models/emotion_model.h5"
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Train model
    model, history, final_metrics = train_model(DATA_DIR, MODEL_SAVE_PATH) 