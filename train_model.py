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
IMG_SIZE = 64    # Increased image size for better feature extraction
EPOCHS = 20      # Increased epochs for better learning
VALIDATION_SPLIT = 0.2

def prepare_dataset():
    # Create train and validation directories
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/validation', exist_ok=True)
    
    rice_types = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']
    
    for rice_type in rice_types:
        os.makedirs(f'dataset/train/{rice_type}', exist_ok=True)
        os.makedirs(f'dataset/validation/{rice_type}', exist_ok=True)
        
        images = glob.glob(f'dataset/{rice_type}/*.jpg')
        if not images:
            raise ValueError(f"No images found for {rice_type}")
            
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=VALIDATION_SPLIT, 
            random_state=42,
            shuffle=True
        )
        
        for img in train_imgs:
            filename = os.path.basename(img)
            shutil.copy2(img, f'dataset/train/{rice_type}/{filename}')
            
        for img in val_imgs:
            filename = os.path.basename(img)
            shutil.copy2(img, f'dataset/validation/{rice_type}/{filename}')
    
    return len(glob.glob('dataset/train/*/*.jpg')), len(glob.glob('dataset/validation/*/*.jpg'))

def create_model():
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
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    
    # Use Adam optimizer with a lower learning rate
    optimizer = Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    train_count, val_count = prepare_dataset()
    
    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    model = create_model()
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Calculate steps
    steps_per_epoch = train_count // BATCH_SIZE
    validation_steps = val_count // BATCH_SIZE
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping]
    )
    
    model.save('rice_classifier.h5')
    return history

if __name__ == "__main__":
    history = train_model()
