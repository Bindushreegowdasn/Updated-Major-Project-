from logger_setup import Logger
import sys

sys.stdout = Logger('project_log.txt')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================
# 1. CONFIGURATION
# =============================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Update these paths to your dataset
TRAIN_DIR = 'dataset/train'
VALIDATION_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# =============================================
# 2. DATA AUGMENTATION (Key for Accuracy!)
# =============================================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,  # Rotate images
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip horizontally
    vertical_flip=True,  # Flip vertically (good for leaves)
    brightness_range=[0.8, 1.2],  # Adjust brightness
    fill_mode='nearest'
)

# Validation and test data - only rescale, no augmentation
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# =============================================
# 3. LOAD DATA
# =============================================
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("\nLoading validation data...")
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\nLoading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\nNumber of classes: {num_classes}")
print(f"Class names: {list(train_generator.class_indices.keys())}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")


# =============================================
# PLOTTING FUNCTION
# =============================================
def plot_accuracy_comparison(train_acc, val_acc, phase_name):
    """Plot accuracy comparison for training and validation"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)

    plt.title(f'Accuracy Progress - {phase_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add max accuracy annotation
    max_val_acc = max(val_acc)
    max_val_epoch = val_acc.index(max_val_acc) + 1
    plt.annotate(f'Max Val Acc: {max_val_acc:.4f}\nEpoch: {max_val_epoch}',
                 xy=(max_val_epoch, max_val_acc),
                 xytext=(max_val_epoch + 1, max_val_acc - 0.1),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'accuracy_{phase_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n{phase_name} - Maximum Validation Accuracy: {max_val_acc:.4f} at Epoch {max_val_epoch}")


# =============================================
# 4. MODEL ARCHITECTURE - TRANSFER LEARNING
# =============================================
def create_transfer_learning_model(num_classes, model_type='mobilenet'):
    """
    Create a model using transfer learning.
    Options: 'mobilenet', 'efficientnet', 'resnet50'
    """

    # Load pre-trained base model
    if model_type == 'mobilenet':
        base_model = MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_type == 'efficientnet':
        base_model = EfficientNetB0(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )

    # Freeze the base model initially
    base_model.trainable = False

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model


# =============================================
# 5. BUILD AND COMPILE MODEL
# =============================================
print("\n" + "=" * 70)
print("Building model with Transfer Learning (MobileNetV2)...")
print("=" * 70)

model, base_model = create_transfer_learning_model(num_classes, model_type='mobilenet')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================================
# 6. CALLBACKS FOR BETTER TRAINING
# =============================================
callbacks = [
    # Save the best model
    ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Stop if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate if plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# =============================================
# 7. PHASE 1: TRAIN TOP LAYERS ONLY
# =============================================
print("\n" + "=" * 70)
print("PHASE 1: Training top layers with frozen base model...")
print("=" * 70)

history1 = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Plot Phase 1 Accuracy
plot_accuracy_comparison(history1.history['accuracy'],
                         history1.history['val_accuracy'],
                         "Phase 1 - Frozen Base Model")

# =============================================
# 8. PHASE 2: FINE-TUNING (Unfreeze some layers)
# =============================================
print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning - Unfreezing last layers of base model...")
print("=" * 70)

# Unfreeze the last 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=len(history1.history['accuracy']),
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Plot Phase 2 Accuracy
plot_accuracy_comparison(history2.history['accuracy'],
                         history2.history['val_accuracy'],
                         "Phase 2 - Fine-tuning")


# =============================================
# 9. COMBINED ACCURACY PLOT
# =============================================
def plot_combined_accuracy(history1, history2):
    """Plot combined accuracy for both phases"""
    # Combine histories
    train_acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

    epochs_range = range(1, len(train_acc) + 1)
    phase_split = len(history1.history['accuracy'])

    plt.figure(figsize=(12, 8))

    # Plot accuracies
    plt.plot(epochs_range, train_acc, 'b-', label='Training Accuracy', linewidth=2, alpha=0.7)
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.7)

    # Add phase separation line
    plt.axvline(x=phase_split, color='green', linestyle='--', linewidth=2,
                label='Fine-tuning Start', alpha=0.8)

    plt.title('Complete Training - Accuracy Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add annotations for best accuracies
    max_phase1_val = max(history1.history['val_accuracy'])
    max_phase1_epoch = history1.history['val_accuracy'].index(max_phase1_val) + 1

    max_phase2_val = max(history2.history['val_accuracy'])
    max_phase2_epoch = history2.history['val_accuracy'].index(max_phase2_val) + phase_split + 1

    plt.annotate(f'Phase 1 Best: {max_phase1_val:.4f}',
                 xy=(max_phase1_epoch, max_phase1_val),
                 xytext=(max_phase1_epoch + 2, max_phase1_val - 0.15),
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontweight='bold')

    plt.annotate(f'Phase 2 Best: {max_phase2_val:.4f}',
                 xy=(max_phase2_epoch, max_phase2_val),
                 xytext=(max_phase2_epoch + 2, max_phase2_val - 0.15),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontweight='bold')

    # Fill between phases
    plt.axvspan(0, phase_split, alpha=0.1, color='blue', label='Phase 1: Frozen Base')
    plt.axvspan(phase_split, len(train_acc), alpha=0.1, color='red', label='Phase 2: Fine-tuning')

    plt.tight_layout()
    plt.savefig('complete_training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nCOMPLETE TRAINING SUMMARY:")
    print(f"Phase 1 Best Validation Accuracy: {max_phase1_val:.4f}")
    print(f"Phase 2 Best Validation Accuracy: {max_phase2_val:.4f}")
    print(f"Overall Best Validation Accuracy: {max(max_phase1_val, max_phase2_val):.4f}")


# Plot combined accuracy
plot_combined_accuracy(history1, history2)

# =============================================
# 10. EVALUATE ON TEST SET
# =============================================
print("\n" + "=" * 70)
print("Evaluating on test set...")
print("=" * 70)

# Load best model
best_model = keras.models.load_model('models/best_model.h5')

# Evaluate
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# =============================================
# 11. FINAL ACCURACY VISUALIZATION
# =============================================
def plot_final_results(train_acc, val_acc, test_accuracy):
    """Plot final results comparison"""
    labels = ['Training', 'Validation', 'Test']
    accuracies = [max(train_acc) if train_acc else 0,
                  max(val_acc) if val_acc else 0,
                  test_accuracy]

    colors = ['blue', 'orange', 'green']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')

    plt.title('Final Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)

    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# Plot final results
all_train_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
plot_final_results(all_train_acc, all_val_acc, test_accuracy)

# =============================================
# 12. DETAILED PREDICTIONS & METRICS
# =============================================
print("\nGenerating predictions...")
predictions = best_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT:")
print("=" * 70)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")


# =============================================
# 13. PLOT TRAINING HISTORY (Original)
# =============================================
def plot_training_history(history1, history2):
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--',
                label='Fine-tuning starts')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--',
                label='Fine-tuning starts')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.show()


plot_training_history(history1, history2)

# =============================================
# 14. SAVE FINAL MODEL
# =============================================
best_model.save('models/final_medicinal_plant_model.h5')
print("\nFinal model saved as 'models/final_medicinal_plant_model.h5'")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("\nAccuracy graphs generated:")
print("- accuracy_phase_1_frozen_base_model.png")
print("- accuracy_phase_2_fine_tuning.png")
print("- complete_training_accuracy.png")
print("- final_accuracy_comparison.png")
print("- training_history.png")
print("=" * 70)