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
from sklearn.utils.class_weight import compute_class_weight
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
# 3.5: CALCULATE CLASS WEIGHTS (Handle Imbalance)
# =============================================
print("\n" + "=" * 70)
print("Calculating class weights to handle imbalanced dataset...")
print("=" * 70)

# Compute class weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

# Convert to dictionary format
class_weight_dict = dict(enumerate(class_weights_array))

print("\nClass weights calculated:")
for class_idx, weight in class_weight_dict.items():
    class_name = list(train_generator.class_indices.keys())[class_idx]
    print(f"  {class_name}: {weight:.2f}")

print("\n✅ Classes with fewer images will get higher weights during training!")


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
    class_weight=class_weight_dict,  # ✅ ADDED: Handle imbalanced classes
    verbose=1
)

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
    initial_epoch=history1.epoch[-1],
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,  # ✅ ADDED: Handle imbalanced classes
    verbose=1
)

# =============================================
# 9. EVALUATE ON TEST SET
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
# 10. DETAILED PREDICTIONS & METRICS
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
# 11. PLOT TRAINING HISTORY
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
# 12. SAVE FINAL MODEL
# =============================================
best_model.save('models/final_medicinal_plant_model.h5')
print("\nFinal model saved as 'models/final_medicinal_plant_model.h5'")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("\nIf accuracy is still below 90%, consider:")
print("- Collecting more training images (100+ per class)")
print("- Checking for mislabeled images")
print("- Trying EfficientNetB0 instead of MobileNetV2")
print("- Adjusting augmentation parameters")
print("=" * 70)