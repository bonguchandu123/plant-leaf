import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Root directory (plant-leaf/)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Backend directory (plant-leaf/backend/)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

# Models directory (plant-leaf/backend/models/)
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

# Dataset directory (plant-leaf/dataset/)
DATASET_PATH = os.path.join(ROOT_DIR, "dataset")

print("=" * 60)
print("PATH CONFIGURATION")
print("=" * 60)
print(f"Root Directory: {ROOT_DIR}")
print(f"Backend Directory: {BACKEND_DIR}")
print(f"Models Directory: {MODELS_DIR}")
print(f"Dataset Directory: {DATASET_PATH}")
print("=" * 60)

# Create backend and models directories if they don't exist
os.makedirs(BACKEND_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# FIX BATCHNORMALIZATION AXIS ISSUE
# ============================================================================
print("\nApplying BatchNormalization axis fix...")

# Store original BatchNormalization __init__
original_bn_init = tf.keras.layers.BatchNormalization.__init__

def fixed_bn_init(self, axis=-1, **kwargs):
    """Fixed BatchNormalization that converts list axis to int"""
    # Convert list to int if needed
    if isinstance(axis, list):
        if len(axis) == 1:
            axis = axis[0]
        else:
            # For multiple axes, keep as tuple
            axis = tuple(axis)
    original_bn_init(self, axis=axis, **kwargs)

# Apply the fix
tf.keras.layers.BatchNormalization.__init__ = fixed_bn_init
print("✅ BatchNormalization fix applied")

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# Output paths (all in backend/models/)
MODEL_H5_PATH = os.path.join(MODELS_DIR, "plant_disease_model.h5")
MODEL_KERAS_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.pkl")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.txt")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.txt")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model_checkpoint.h5")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.png")
MANIFEST_PATH = os.path.join(ROOT_DIR, "dataset_manifest.csv")

print("\n" + "=" * 60)
print("Plant Disease Detection - Model Training")
print("=" * 60)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)


def discover_classes():
    """
    Automatically discover all disease classes from dataset folder
    """
    print("\n[0/6] Discovering disease classes from dataset folder...")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found!")
    
    # Get all subdirectories (each is a class)
    classes = []
    for item in sorted(os.listdir(DATASET_PATH)):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            # Check if folder has images
            images = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > 0:
                classes.append(item)
    
    if len(classes) == 0:
        raise ValueError("No valid disease classes found in dataset folder!")
    
    print(f"\n✓ Found {len(classes)} disease classes:")
    for i, cls in enumerate(classes, 1):
        print(f"  {i}. {cls}")
    
    return classes


def load_dataset(classes):
    """
    Load images from dataset folder and create train/val/test splits
    """
    print("\n[1/6] Loading dataset...")
    
    image_paths = []
    labels = []
    
    # Load all images and labels
    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found! Skipping...")
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(class_name)
        
        print(f"  - {class_name}: {len(images)} images")
    
    print(f"\n✓ Total images loaded: {len(image_paths)}")
    
    # Create manifest CSV
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"✓ Dataset manifest saved to: {MANIFEST_PATH}")
    
    return image_paths, labels


def preprocess_and_split(image_paths, labels):
    """
    Split dataset into train, validation, and test sets
    """
    print("\n[2/6] Splitting dataset (70% train, 15% val, 15% test)...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    
    # Second split: 15% val, 15% test (50-50 split of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


def create_data_generators(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Create data generators with augmentation for training
    """
    print("\n[3/6] Creating data generators with augmentation...")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Helper function to load and preprocess images
    def load_and_preprocess(paths, labels):
        images = []
        valid_labels = []
        for i, path in enumerate(paths):
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                valid_labels.append(labels[i])
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
        return np.array(images), np.array(valid_labels)
    
    # Load images
    print("  - Loading training images...")
    X_train_imgs, y_train_arr = load_and_preprocess(X_train, y_train)
    print(f"    Loaded: {len(X_train_imgs)} images")
    
    print("  - Loading validation images...")
    X_val_imgs, y_val_arr = load_and_preprocess(X_val, y_val)
    print(f"    Loaded: {len(X_val_imgs)} images")
    
    print("  - Loading test images...")
    X_test_imgs, y_test_arr = load_and_preprocess(X_test, y_test)
    print(f"    Loaded: {len(X_test_imgs)} images")
    
    print("✓ Images loaded and preprocessed successfully")
    
    return (X_train_imgs, y_train_arr), (X_val_imgs, y_val_arr), (X_test_imgs, y_test_arr), train_datagen


def build_model(num_classes):
    """
    Build CNN model using MobileNetV2 as base with transfer learning
    """
    print(f"\n[4/6] Building model architecture (MobileNetV2 + Custom Layers)...")
    print(f"  - Number of output classes: {num_classes}")
    
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build custom top layers WITHOUT axis parameter in BatchNormalization
    # After GlobalAveragePooling2D, the tensor is 2D, so we use default axis=-1
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),  # ✅ No axis parameter - uses default axis=-1
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),  # ✅ No axis parameter - uses default axis=-1
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model architecture created")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    return model


def train_model(model, train_data, val_data, train_datagen):
    """
    Train the model with callbacks
    """
    print("\n[5/6] Training model...")
    print("-" * 60)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Fit data generator to training data
    train_datagen.fit(X_train)
    
    # Train model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    print("-" * 60)
    print("✓ Training completed!")
    
    return model, history


def evaluate_model(model, test_data, label_encoder):
    """
    Evaluate model on test set and display metrics
    """
    print("\n[6/6] Evaluating model on test set...")
    
    X_test, y_test = test_data
    
    # Predictions
    print("  - Making predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    print("\n  - Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Adjust figure size based on number of classes
    num_classes = len(label_encoder.classes_)
    fig_size = max(10, num_classes * 0.8)
    
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
    print(f"  ✓ Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    
    # Test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'='*60}")


def plot_training_history(history):
    """
    Plot training history (accuracy and loss)
    """
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to: {TRAINING_HISTORY_PATH}")


def save_model(model, label_encoder):
    """
    Save model and label encoder in multiple formats
    """
    print(f"\nSaving model to backend/models/...")
    
    # 1. Save as HDF5 (H5) - Most compatible
    print(f"  - Saving as H5: {MODEL_H5_PATH}")
    model.save(MODEL_H5_PATH, save_format='h5')
    print("    ✓ H5 format saved")
    
    # 2. Save as Keras format (recommended for TF 2.x)
    print(f"  - Saving as Keras: {MODEL_KERAS_PATH}")
    model.save(MODEL_KERAS_PATH)
    print("    ✓ Keras format saved")
    
    # 3. Save as PKL (with label encoder)
    print(f"  - Saving as PKL: {MODEL_PKL_PATH}")
    model_package = {
        'model': model,
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_.tolist(),
        'img_size': IMG_SIZE
    }
    joblib.dump(model_package, MODEL_PKL_PATH)
    print("    ✓ PKL format saved")
    
    # 4. Save class names separately for easy reference
    print(f"  - Saving class names: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, 'w', encoding='utf-8') as f:
        for cls in label_encoder.classes_:
            f.write(f"{cls}\n")
    print("    ✓ Class names saved")
    
    # 5. Save model metadata
    print(f"  - Saving metadata: {METADATA_PATH}")
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Model Training Metadata\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Number of Classes: {len(label_encoder.classes_)}\n")
        f.write(f"Total Parameters: {model.count_params():,}\n")
        f.write(f"\nClasses:\n")
        for i, cls in enumerate(label_encoder.classes_, 1):
            f.write(f"  {i}. {cls}\n")
    print("    ✓ Metadata saved")
    
    print("\n✓ All model files saved successfully!")


def main():
    """
    Main training pipeline
    """
    try:
        print("\n" + "="*60)
        print("STARTING TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Verify directories exist
        if not os.path.exists(DATASET_PATH):
            print(f"\n❌ Error: Dataset folder not found at: {DATASET_PATH}")
            print("\nExpected structure:")
            print("plant-leaf/")
            print("  ├── backend/")
            print("  │   ├── main.py")
            print("  │   └── models/  (will be created)")
            print("  ├── dataset/")
            print("  │   ├── Disease_Class_1/")
            print("  │   │   ├── image1.jpg")
            print("  │   │   └── ...")
            print("  │   └── Disease_Class_2/")
            print("  └── train_model.py (this file)")
            return
        
        # Step 0: Discover all disease classes
        classes = discover_classes()
        
        # Step 1: Load dataset
        image_paths, labels = load_dataset(classes)
        
        if len(image_paths) == 0:
            print("\n❌ Error: No images found! Please check dataset folder structure.")
            return
        
        # Step 2: Preprocess and split
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = preprocess_and_split(image_paths, labels)
        
        # Step 3: Create data generators
        train_data, val_data, test_data, train_datagen = create_data_generators(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 4: Build model
        model = build_model(num_classes=len(classes))
        
        # Step 5: Train model
        model, history = train_model(model, train_data, val_data, train_datagen)
        
        # Step 6: Evaluate model
        evaluate_model(model, test_data, label_encoder)
        
        # Plot training history
        plot_training_history(history)
        
        # Save model
        save_model(model, label_encoder)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"✓ Total classes trained: {len(classes)}")
        print(f"✓ Model files saved in: {MODELS_DIR}")
        print(f"  - plant_disease_model.h5")
        print(f"  - plant_disease_model.keras")
        print(f"  - plant_disease_model.pkl")
        print(f"  - class_names.txt")
        print(f"  - model_metadata.txt")
        print(f"  - confusion_matrix.png")
        print(f"  - training_history.png")
        print(f"  - best_model_checkpoint.h5")
        print(f"\n✓ You can now use this model in your FastAPI backend!")
        print(f"✓ Update config.py MODEL_PATH to: ./models/plant_disease_model.h5")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("  1. Dataset folder structure is correct")
        print("  2. All dependencies are installed")
        print("  3. Sufficient disk space available")


if __name__ == "__main__":
    main()