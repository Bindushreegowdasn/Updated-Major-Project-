from logger_setup import Logger
import sys

sys.stdout = Logger('project_log.txt')


import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class ModelEvaluator:
    def __init__(self, model_path='models/best_model.h5'):
        print("Loading model for evaluation...")
        self.model = keras.models.load_model(model_path)
        self.img_size = (224, 224)
        self.class_names = self.get_class_names()
        print(f"Classes: {self.class_names}")

    def get_class_names(self):
        """Get class names from training directory"""
        train_dir = 'dataset/train'
        if os.path.exists(train_dir):
            classes = sorted([d for d in os.listdir(train_dir)
                              if os.path.isdir(os.path.join(train_dir, d))])
            return classes
        return []

    def evaluate_single_image(self, image_path):
        """Evaluate a single image with detailed analysis"""
        actual_class = os.path.basename(os.path.dirname(image_path))

        # Preprocess image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = self.model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = predictions[predicted_idx]

        # Create detailed results
        results = {
            'actual_class': actual_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_correct': actual_class == predicted_class,
            'all_predictions': []
        }

        # Get all predictions sorted
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices:
            results['all_predictions'].append({
                'class': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'confidence_percent': f"{predictions[idx] * 100:.2f}%"
            })

        return results, img

    def evaluate_multiple_images(self, num_images=5):
        """Evaluate multiple random images from validation set"""
        validation_dir = 'dataset/validation'
        results = []

        for class_name in self.class_names:
            class_path = os.path.join(validation_dir, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                # Test 1-2 images per class
                for img_name in images[:min(2, len(images))]:
                    img_path = os.path.join(class_path, img_name)
                    result, _ = self.evaluate_single_image(img_path)
                    results.append(result)

        return results

    def plot_predictions(self, results, image):
        """Plot the image with prediction results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot image
        ax1.imshow(image)
        ax1.set_title(f"Actual: {results['actual_class']}\nPredicted: {results['predicted_class']}")
        ax1.axis('off')

        # Plot confidence scores
        classes = [p['class'] for p in results['all_predictions'][:8]]  # Top 8
        confidences = [p['confidence'] for p in results['all_predictions'][:8]]
        colors = ['red' if c == results['predicted_class'] else
                  'green' if c == results['actual_class'] else 'blue'
                  for c in classes]

        bars = ax2.barh(classes, confidences, color=colors, alpha=0.7)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Prediction Confidence Scores')
        ax2.set_xlim(0, 1)

        # Add value labels on bars
        for bar, confidence in zip(bars, confidences):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{confidence:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    evaluator = ModelEvaluator()

    # Test with the same image
    image_path = "dataset/validation/Arive-Dantu/AV-S-001.jpg"

    print("🧪 Testing single image...")
    results, image = evaluator.evaluate_single_image(image_path)

    print("\n" + "=" * 70)
    print("📊 DETAILED PREDICTION ANALYSIS")
    print("=" * 70)
    print(f"🖼️  Image: {os.path.basename(image_path)}")
    print(f"📂 Actual Class: {results['actual_class']}")
    print(f"🔮 Predicted Class: {results['predicted_class']}")
    print(f"🎯 Confidence: {results['confidence']:.4f} ({results['confidence'] * 100:.2f}%)")
    print(f"✅ Correct: {results['is_correct']}")
    print("=" * 70)

    print("\n📈 ALL PREDICTIONS (Sorted by Confidence):")
    for i, pred in enumerate(results['all_predictions'][:10], 1):
        marker = "🎯" if pred['class'] == results['predicted_class'] else "📌"
        actual_marker = "✅" if pred['class'] == results['actual_class'] else ""
        print(f"{i:2d}. {marker} {pred['class']:20} {pred['confidence_percent']} {actual_marker}")

    # Plot results
    evaluator.plot_predictions(results, image)

    # Test multiple images
    print("\n" + "=" * 70)
    print("🧪 TESTING MULTIPLE IMAGES...")
    print("=" * 70)

    multiple_results = evaluator.evaluate_multiple_images()
    correct_predictions = sum(1 for r in multiple_results if r['is_correct'])
    accuracy = correct_predictions / len(multiple_results) if multiple_results else 0

    print(f"\n📊 Multiple Image Test Results:")
    print(f"   Images tested: {len(multiple_results)}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Show some examples
    print(f"\n🔍 Sample predictions:")
    for i, result in enumerate(multiple_results[:3]):
        status = "✅ CORRECT" if result['is_correct'] else "❌ WRONG"
        print(
            f"   {i + 1}. {result['actual_class']} → {result['predicted_class']} ({result['confidence']:.3f}) - {status}")


if __name__ == "__main__":
    main()