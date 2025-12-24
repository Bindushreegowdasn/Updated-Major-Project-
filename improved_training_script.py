
# IMPROVED TRAINING SCRIPT FOR MEDICINAL PLANTS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import os
import numpy as np

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,  # Increased from 40
    width_shift_range=0.3,  # Increased from 0.2
    height_shift_range=0.3,
    shear_range=0.3,  # Increased from 0.2
    zoom_range=0.3,   # Increased from 0.2
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],  # Wider range
    channel_shift_range=0.3,  # Added color variation
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Use EfficientNetB0 for better feature extraction
def create_improved_model(num_classes):
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.6),  # Increased dropout
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model

# Create model
model, base_model = create_improved_model(num_classes)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint(
        'models/improved_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

print("Starting improved training...")

# Phase 1: Train with frozen base
history1 = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,  # Use class weights
    verbose=1
)

# Phase 2: Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-40]:  # Unfreeze more layers
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=50,
    initial_epoch=len(history1.history['accuracy']),
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("Improved training completed!")
print("Use the model for better predictions on Arive-Dantu and similar plants.")
