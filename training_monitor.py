import os
import time
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from PIL import Image


def monitor_training_progress():
    """Monitor the training progress and test improvements"""

    # Wait for training to complete or check intermediate results
    model_path = 'models/improved_model.h5'

    if os.path.exists(model_path):
        print("Loading improved model for testing...")
        model = keras.models.load_model(model_path)

        # Test with the same Arive-Dantu image
        test_image = "dataset/validation/Arive-Dantu/AV-S-001.jpg"

        if os.path.exists(test_image):
            # Load and preprocess
            img = Image.open(test_image)
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)[0]
            predicted_idx = np.argmax(predictions)

            # Get class names from training directory
            train_dir = 'dataset/train'
            class_names = sorted([d for d in os.listdir(train_dir)
                                  if os.path.isdir(os.path.join(train_dir, d))])

            print("\n" + "=" * 60)
            print("IMPROVED MODEL TEST RESULTS")
            print("=" * 60)

            print(f"Testing Arive-Dantu image: {os.path.basename(test_image)}")
            print(f"Predicted: {class_names[predicted_idx]}")
            print(f"Confidence: {predictions[predicted_idx]:.4f} ({predictions[predicted_idx] * 100:.2f}%)")

            # Show top 5 predictions
            print("\nTop 5 Predictions:")
            sorted_indices = np.argsort(predictions)[::-1][:5]
            for i, idx in enumerate(sorted_indices):
                confidence = predictions[idx]
                status = "✅ ACTUAL" if class_names[idx] == "Arive-Dantu" else ""
                print(f"  {i + 1}. {class_names[idx]:20} {confidence:.4f} ({confidence * 100:.2f}%) {status}")

            # Check if improvement achieved
            if class_names[predicted_idx] == "Arive-Dantu" and predictions[predicted_idx] > 0.7:
                print("\n🎉 SUCCESS: Model now correctly identifies Arive-Dantu with high confidence!")
            elif class_names[predicted_idx] == "Arive-Dantu":
                print("\n⚠️  Model identifies Arive-Dantu but confidence is low")
            else:
                print("\n❌ Model still confused - may need more training/data")

        else:
            print("Test image not found!")
    else:
        print("Improved model not found yet. Training may still be in progress.")
        print("Run this script again after training completes.")


def compare_models():
    """Compare old vs new model performance"""
    old_model_path = 'models/best_model.h5'
    new_model_path = 'models/improved_model.h5'

    if os.path.exists(old_model_path) and os.path.exists(new_model_path):
        print("\n" + "=" * 60)
        print("MODEL COMPARISON: OLD vs NEW")
        print("=" * 60)

        old_model = keras.models.load_model(old_model_path)
        new_model = keras.models.load_model(new_model_path)

        test_image = "dataset/validation/Arive-Dantu/AV-S-001.jpg"

        if os.path.exists(test_image):
            # Load and preprocess
            img = Image.open(test_image)
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Get class names
            train_dir = 'dataset/train'
            class_names = sorted([d for d in os.listdir(train_dir)
                                  if os.path.isdir(os.path.join(train_dir, d))])

            # Old model prediction
            old_pred = old_model.predict(img_array)[0]
            old_idx = np.argmax(old_pred)

            # New model prediction
            new_pred = new_model.predict(img_array)[0]
            new_idx = np.argmax(new_pred)

            print(f"\nTest Image: Arive-Dantu/AV-S-001.jpg")
            print(f"\nOLD MODEL:")
            print(f"  Prediction: {class_names[old_idx]}")
            print(f"  Confidence: {old_pred[old_idx]:.4f} ({old_pred[old_idx] * 100:.2f}%)")
            print(f"  Correct: {'✅' if class_names[old_idx] == 'Arive-Dantu' else '❌'}")

            print(f"\nNEW MODEL:")
            print(f"  Prediction: {class_names[new_idx]}")
            print(f"  Confidence: {new_pred[new_idx]:.4f} ({new_pred[new_idx] * 100:.2f}%)")
            print(f"  Correct: {'✅' if class_names[new_idx] == 'Arive-Dantu' else '❌'}")

            # Calculate improvement
            if class_names[new_idx] == 'Arive-Dantu' and class_names[old_idx] != 'Arive-Dantu':
                print(f"\n🎉 IMPROVEMENT: Model now correctly identifies Arive-Dantu!")
            elif new_pred[new_idx] > old_pred[old_idx] and class_names[new_idx] == 'Arive-Dantu':
                improvement = (new_pred[new_idx] - old_pred[old_idx]) * 100
                print(f"\n📈 IMPROVEMENT: Confidence increased by {improvement:.2f}%")


if __name__ == "__main__":
    monitor_training_progress()
    compare_models()