
from logger_setup import Logger
import sys
sys.stdout = Logger('project_log.txt')


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("=" * 50)
print("BUILDING CNN MODEL")
print("=" * 50)

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 30  # Update this based on your number of plant classes


def build_cnn_model(num_classes):
    """
    Build a Convolutional Neural Network for leaf classification.

    Architecture:
    - 4 Convolutional blocks with MaxPooling
    - Batch Normalization for stable training
    - Dropout for preventing overfitting
    - Dense layers for classification
    """

    print("\nBuilding CNN architecture...")

    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Flatten the feature maps
        layers.Flatten(),

        # Fully Connected Layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Build the model
model = build_cnn_model(NUM_CLASSES)

print("✓ Model architecture created!")

# Display model summary
print("\n" + "=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
model.summary()

# Count total parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# Compile the model
print("\n" + "=" * 50)
print("COMPILING MODEL")
print("=" * 50)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully!")
print("\nOptimizer: Adam")
print("Learning Rate: 0.001")
print("Loss Function: Categorical Crossentropy")
print("Metrics: Accuracy")

# Save model architecture visualization
print("\n" + "=" * 50)
print("SAVING MODEL ARCHITECTURE")
print("=" * 50)

try:
    keras.utils.plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=96
    )
    print("✓ Model architecture diagram saved as 'model_architecture.png'")
except:
    print("⚠ Could not save model diagram (graphviz not installed)")

print("\n" + "=" * 50)
print("MODEL BUILDING COMPLETE!")
print("=" * 50)
print("\nNext step: Train the model!")