import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


class MedicinalPlantPredictor:
    def __init__(self, model_path='models/best_model.h5'):
        print("Loading trained model...")
        self.model = keras.models.load_model(model_path)
        self.img_size = (224, 224)

        # Let's discover the actual class names from your model training
        self.class_names = self.get_actual_class_names()
        print(f"Model loaded! Actual classes: {self.class_names}")

    def get_actual_class_names(self):
        """Get the actual class names used during training"""
        # Check if we can find class names from the dataset
        train_dir = 'dataset/train'
        if os.path.exists(train_dir):
            classes = sorted([d for d in os.listdir(train_dir)
                              if os.path.isdir(os.path.join(train_dir, d))])
            if classes:
                return classes

        # Fallback to our assumed classes
        return ['Aloe Vera', 'Tulsi', 'Neem', 'Turmeric', 'Ginger',
                'Ashwagandha', 'Brahmi', 'Giloy', 'Henna', 'Lemongrass']

    def predict_image(self, image_path):
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = self.model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]

            return {
                'class': predicted_class,
                'confidence': float(confidence),
                'class_index': predicted_class_idx,
                'all_predictions': predictions[0]
            }
        except Exception as e:
            print(f"Error: {e}")
            return None


def main():
    predictor = MedicinalPlantPredictor()

    # Use the exact path we found
    image_path = "dataset/validation/Arive-Dantu/AV-S-001.jpg"

    # Fix path separators for Windows
    image_path = image_path.replace('\\', '/')

    print(f"Testing with image: {image_path}")

    if os.path.exists(image_path):
        result = predictor.predict_image(image_path)

        if result:
            print("\n" + "=" * 60)
            print("🔬 MEDICINAL PLANT PREDICTION RESULTS")
            print("=" * 60)
            print(f"🌿 Predicted Plant: {result['class']}")
            print(f"🎯 Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
            print(f"📊 Class Index: {result['class_index']}")
            print(f"📁 Image: {os.path.basename(image_path)}")
            print("=" * 60)

            # Show top 5 predictions
            print("\n📈 TOP 5 PREDICTIONS:")
            sorted_indices = np.argsort(result['all_predictions'])[::-1]
            for i, idx in enumerate(sorted_indices[:5]):
                confidence = result['all_predictions'][idx]
                print(f"  {i + 1}. {predictor.class_names[idx]}: {confidence:.4f} ({confidence * 100:.2f}%)")

            # Show actual folder name for comparison
            actual_folder = os.path.basename(os.path.dirname(image_path))
            print(f"\n📂 Actual folder (true label): {actual_folder}")
            print(f"✅ Prediction matches folder: {result['class'] == actual_folder}")

        else:
            print("❌ Prediction failed!")
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please check the file path.")


if __name__ == "__main__":
    main()