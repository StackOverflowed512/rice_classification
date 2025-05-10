import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# Dataset parameters
BATCH_SIZE = 32
IMG_SIZE = 128   # Larger size for better detail detection
EPOCHS = 25
VALIDATION_SPLIT = 0.2

def prepare_condition_dataset():
    # Create train and validation directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    os.makedirs('condition_dataset/train', exist_ok=True)
    os.makedirs('condition_dataset/validation', exist_ok=True)
    
    # Classes for rice condition
    conditions = ['broken', 'damaged', 'discolored']
    
    for condition in conditions:
        os.makedirs(f'condition_dataset/train/{condition}', exist_ok=True)
        os.makedirs(f'condition_dataset/validation/{condition}', exist_ok=True)
        
        # Get images from the dataset directory with absolute path
        condition_path = os.path.join(dataset_dir, condition)
        if not os.path.exists(condition_path):
            raise ValueError(f"Directory not found: {condition_path}")
            
        images = []
        for ext in ['jpg', 'jpeg', 'png']:
            images.extend(glob.glob(os.path.join(condition_path, f'*.{ext}')))
            
        if not images:
            raise ValueError(f"No images found in {condition_path}")
            
        print(f"Found {len(images)} images for {condition}")
            
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=VALIDATION_SPLIT, 
            random_state=42,
            shuffle=True
        )
        
        print(f"Splitting {condition}: {len(train_imgs)} training, {len(val_imgs)} validation")
        
        for img in train_imgs:
            filename = os.path.basename(img)
            dst = os.path.join('condition_dataset/train', condition, filename)
            shutil.copy2(img, dst)
            
        for img in val_imgs:
            filename = os.path.basename(img)
            dst = os.path.join('condition_dataset/validation', condition, filename)
            shutil.copy2(img, dst)
    
    train_count = len(glob.glob('condition_dataset/train/*/*.jpg'))
    val_count = len(glob.glob('condition_dataset/validation/*/*.jpg'))
    
    print(f"Total images: {train_count} training, {val_count} validation")
    return train_count, val_count

def create_condition_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: broken, damaged, discolored
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_condition_model():
    train_count, val_count = prepare_condition_dataset()
    
    # Modify data augmentation to be less aggressive
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,        # Reduced from 20
        width_shift_range=0.1,    # Reduced from 0.2
        height_shift_range=0.1,   # Reduced from 0.2
        shear_range=0.1,         # Reduced from 0.2
        zoom_range=0.1,          # Reduced from 0.2
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],  # Reduced range
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'condition_dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        'condition_dataset/validation',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    model = create_condition_model()
    
    # Modified early stopping with increased patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,             # Increased from 5 to 10
        min_delta=0.001,        # Minimum change to count as an improvement
        restore_best_weights=True
    )
    
    # Add learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'condition_classifier.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    history = train_condition_model()