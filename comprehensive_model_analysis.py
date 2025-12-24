import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json


class ModelAnalyzer:
    def __init__(self, model_path, test_dir, class_names=None):
        """
        Initialize the Model Analyzer

        Args:
            model_path: Path to the saved model
            test_dir: Path to test dataset directory
            class_names: List of class names (if None, will be inferred)
        """
        self.model = keras.models.load_model(model_path)
        self.test_dir = test_dir
        self.class_names = class_names
        self.predictions = None
        self.true_labels = None
        self.confidence_scores = None

    def load_and_predict(self, img_size=(224, 224), batch_size=32):
        """Load test data and make predictions"""
        # Create test data generator
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        if self.class_names is None:
            self.class_names = list(self.test_generator.class_indices.keys())

        print("Making predictions on test set...")
        predictions = self.model.predict(self.test_generator, verbose=1)

        self.confidence_scores = predictions
        self.predictions = np.argmax(predictions, axis=1)
        self.true_labels = self.test_generator.classes

        return predictions

    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        """Create and save confusion matrix"""
        cm = confusion_matrix(self.true_labels, self.predictions)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Medicinal Plant Classification', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()

    def plot_normalized_confusion_matrix(self, save_path='confusion_matrix_normalized.png'):
        """Create normalized confusion matrix (percentages)"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix (Accuracy per Class)', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Normalized confusion matrix saved to {save_path}")
        plt.close()

    def generate_classification_report(self, save_path='classification_report.txt'):
        """Generate detailed classification report"""
        report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            digits=4
        )

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CLASSIFICATION REPORT - MEDICINAL PLANT CLASSIFICATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Overall Accuracy: {accuracy_score(self.true_labels, self.predictions):.4f}\n")
            f.write("=" * 70 + "\n")

        print(f"✓ Classification report saved to {save_path}")
        print("\n" + report)

    def analyze_per_class_performance(self, save_path='per_class_analysis.png'):
        """Analyze performance for each class"""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions
        )

        # Create DataFrame for better visualization
        df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })

        # Sort by F1-Score
        df = df.sort_values('F1-Score', ascending=True)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Precision
        axes[0].barh(df['Class'], df['Precision'], color='skyblue')
        axes[0].set_xlabel('Precision', fontsize=12)
        axes[0].set_title('Precision by Class', fontsize=14)
        axes[0].set_xlim(0, 1)
        axes[0].axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Target: 0.8')
        axes[0].legend()

        # Recall
        axes[1].barh(df['Class'], df['Recall'], color='lightcoral')
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_title('Recall by Class', fontsize=14)
        axes[1].set_xlim(0, 1)
        axes[1].axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Target: 0.8')
        axes[1].legend()

        # F1-Score
        axes[2].barh(df['Class'], df['F1-Score'], color='lightgreen')
        axes[2].set_xlabel('F1-Score', fontsize=12)
        axes[2].set_title('F1-Score by Class', fontsize=14)
        axes[2].set_xlim(0, 1)
        axes[2].axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Target: 0.8')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class analysis saved to {save_path}")
        plt.close()

        return df

    def find_most_confused_pairs(self, top_n=5, save_path='confused_pairs.txt'):
        """Identify which classes are most commonly confused"""
        cm = confusion_matrix(self.true_labels, self.predictions)

        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append({
                        'True': self.class_names[i],
                        'Predicted': self.class_names[j],
                        'Count': cm[i][j],
                        'Percentage': (cm[i][j] / cm[i].sum()) * 100
                    })

        # Sort by count
        confused_pairs.sort(key=lambda x: x['Count'], reverse=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TOP CONFUSED CLASS PAIRS\n")
            f.write("=" * 80 + "\n\n")

            for idx, pair in enumerate(confused_pairs[:top_n], 1):
                f.write(f"{idx}. True: {pair['True']} -> Predicted: {pair['Predicted']}\n")
                f.write(f"   Count: {pair['Count']} | Percentage: {pair['Percentage']:.2f}%\n\n")

        print(f"✓ Confused pairs analysis saved to {save_path}")
        print("\nTop 5 Most Confused Pairs:")
        for idx, pair in enumerate(confused_pairs[:top_n], 1):
            print(f"{idx}. {pair['True']} -> {pair['Predicted']}: {pair['Count']} times ({pair['Percentage']:.2f}%)")

        return confused_pairs

    def analyze_confidence_distribution(self, save_path='confidence_distribution.png'):
        """Analyze confidence score distribution"""
        max_confidences = np.max(self.confidence_scores, axis=1)
        correct_mask = self.predictions == self.true_labels

        correct_confidences = max_confidences[correct_mask]
        incorrect_confidences = max_confidences[~correct_mask]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Overall distribution
        axes[0, 0].hist(max_confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=np.mean(max_confidences), color='red', linestyle='--',
                           label=f'Mean: {np.mean(max_confidences):.3f}')
        axes[0, 0].set_xlabel('Confidence Score', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Overall Confidence Distribution', fontsize=13)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Correct vs Incorrect
        axes[0, 1].hist([correct_confidences, incorrect_confidences],
                        bins=30, label=['Correct', 'Incorrect'],
                        color=['green', 'red'], alpha=0.6, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence Score', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Confidence: Correct vs Incorrect Predictions', fontsize=13)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Box plot
        data_to_plot = [correct_confidences, incorrect_confidences]
        axes[1, 0].boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                           patch_artist=True,
                           boxprops=dict(facecolor='lightblue'))
        axes[1, 0].set_ylabel('Confidence Score', fontsize=11)
        axes[1, 0].set_title('Confidence Score Distribution (Box Plot)', fontsize=13)
        axes[1, 0].grid(alpha=0.3)

        # Statistics table
        stats_text = f"""
        Confidence Statistics:

        Overall:
          Mean: {np.mean(max_confidences):.4f}
          Median: {np.median(max_confidences):.4f}
          Std: {np.std(max_confidences):.4f}

        Correct Predictions:
          Mean: {np.mean(correct_confidences):.4f}
          Median: {np.median(correct_confidences):.4f}
          Count: {len(correct_confidences)}

        Incorrect Predictions:
          Mean: {np.mean(incorrect_confidences):.4f}
          Median: {np.median(incorrect_confidences):.4f}
          Count: {len(incorrect_confidences)}

        Low Confidence (<0.5): {np.sum(max_confidences < 0.5)}
        High Confidence (>0.9): {np.sum(max_confidences > 0.9)}
        """

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10,
                        verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confidence distribution analysis saved to {save_path}")
        plt.close()

        return {
            'mean_overall': float(np.mean(max_confidences)),
            'mean_correct': float(np.mean(correct_confidences)),
            'mean_incorrect': float(np.mean(incorrect_confidences)),
            'low_confidence_count': int(np.sum(max_confidences < 0.5))
        }

    def identify_difficult_samples(self, top_n=10, save_path='difficult_samples.json'):
        """Identify the most difficult samples (low confidence correct predictions)"""
        max_confidences = np.max(self.confidence_scores, axis=1)
        correct_mask = self.predictions == self.true_labels

        # Get indices of correct but low confidence predictions
        correct_indices = np.where(correct_mask)[0]
        correct_confidences = max_confidences[correct_indices]

        # Sort by confidence (ascending)
        sorted_indices = correct_indices[np.argsort(correct_confidences)]

        difficult_samples = []
        for idx in sorted_indices[:top_n]:
            difficult_samples.append({
                'image_index': int(idx),
                'true_class': self.class_names[self.true_labels[idx]],
                'predicted_class': self.class_names[self.predictions[idx]],
                'confidence': float(max_confidences[idx]),
                'all_scores': {self.class_names[i]: float(self.confidence_scores[idx][i])
                               for i in range(len(self.class_names))}
            })

        with open(save_path, 'w') as f:
            json.dump(difficult_samples, f, indent=2)

        print(f"✓ Difficult samples analysis saved to {save_path}")
        print("\nTop 5 Difficult Samples (Correct but Low Confidence):")
        for i, sample in enumerate(difficult_samples[:5], 1):
            print(f"{i}. Class: {sample['true_class']}, Confidence: {sample['confidence']:.4f}")

        return difficult_samples

    def generate_summary_report(self, save_path='analysis_summary.txt'):
        """Generate a comprehensive summary report"""
        accuracy = accuracy_score(self.true_labels, self.predictions)
        max_confidences = np.max(self.confidence_scores, axis=1)

        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL ANALYSIS SUMMARY REPORT\n")
            f.write("Medicinal Plant Classification by Visual Characteristics\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Test Samples: {len(self.true_labels)}\n")
            f.write(f"Number of Classes: {len(self.class_names)}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
            f.write(f"Correct Predictions: {np.sum(self.predictions == self.true_labels)}\n")
            f.write(f"Incorrect Predictions: {np.sum(self.predictions != self.true_labels)}\n\n")

            f.write("CONFIDENCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Confidence: {np.mean(max_confidences):.4f}\n")
            f.write(f"Median Confidence: {np.median(max_confidences):.4f}\n")
            f.write(f"Min Confidence: {np.min(max_confidences):.4f}\n")
            f.write(f"Max Confidence: {np.max(max_confidences):.4f}\n")
            f.write(f"Samples with Confidence < 0.5: {np.sum(max_confidences < 0.5)}\n")
            f.write(f"Samples with Confidence > 0.9: {np.sum(max_confidences > 0.9)}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if accuracy < 0.8:
                f.write("⚠ Accuracy is below 80%. Consider:\n")
                f.write("  - Adding more training data\n")
                f.write("  - Increasing data augmentation\n")
                f.write("  - Training for more epochs\n")
                f.write("  - Using a more powerful model architecture\n\n")

            if np.mean(max_confidences) < 0.7:
                f.write("⚠ Low confidence scores. Consider:\n")
                f.write("  - Reviewing data quality and labels\n")
                f.write("  - Checking for class imbalance\n")
                f.write("  - Fine-tuning the model further\n\n")

            if np.sum(max_confidences < 0.5) > len(self.true_labels) * 0.1:
                f.write("⚠ Many low-confidence predictions (>10%). Consider:\n")
                f.write("  - Investigating confused classes\n")
                f.write("  - Adding more distinctive features\n")
                f.write("  - Collecting better quality images\n\n")

        print(f"✓ Summary report saved to {save_path}")

    def run_complete_analysis(self, output_dir='analysis_results'):
        """Run all analyses and save results"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE MODEL ANALYSIS")
        print("=" * 80 + "\n")

        # Load and predict if not already done
        if self.predictions is None:
            self.load_and_predict()

        # Run all analyses
        print("\n1. Generating confusion matrices...")
        self.plot_confusion_matrix(f'{output_dir}/confusion_matrix.png')
        self.plot_normalized_confusion_matrix(f'{output_dir}/confusion_matrix_normalized.png')

        print("\n2. Generating classification report...")
        self.generate_classification_report(f'{output_dir}/classification_report.txt')

        print("\n3. Analyzing per-class performance...")
        df = self.analyze_per_class_performance(f'{output_dir}/per_class_analysis.png')
        df.to_csv(f'{output_dir}/per_class_metrics.csv', index=False)

        print("\n4. Finding confused class pairs...")
        self.find_most_confused_pairs(save_path=f'{output_dir}/confused_pairs.txt')

        print("\n5. Analyzing confidence distribution...")
        self.analyze_confidence_distribution(f'{output_dir}/confidence_distribution.png')

        print("\n6. Identifying difficult samples...")
        self.identify_difficult_samples(save_path=f'{output_dir}/difficult_samples.json')

        print("\n7. Generating summary report...")
        self.generate_summary_report(f'{output_dir}/analysis_summary.txt')

        print("\n" + "=" * 80)
        print(f"✓ ANALYSIS COMPLETE! All results saved to '{output_dir}/' directory")
        print("=" * 80 + "\n")


# USAGE
if __name__ == "__main__":
    # Configuration - UPDATED WITH CORRECT PATH
    MODEL_PATH = 'models/best_model.h5'
    TEST_DIR = 'dataset/test'

    # Create analyzer
    analyzer = ModelAnalyzer(
        model_path=MODEL_PATH,
        test_dir=TEST_DIR
    )

    # Run complete analysis
    analyzer.run_complete_analysis(output_dir='analysis_results')