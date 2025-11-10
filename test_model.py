import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

print("Loading trained model...")
print("=" * 50)

# Check if model files exist
model_json_path = "signlanguagedetectionmodel48x48.json"
model_weights_path = "signlanguagedetectionmodel48x48.h5"

if not os.path.exists(model_json_path):
    raise FileNotFoundError(f"Model JSON file not found: {model_json_path}. Please train the model first.")
if not os.path.exists(model_weights_path):
    raise FileNotFoundError(f"Model weights file not found: {model_weights_path}. Please train the model first.")

# Load model architecture
try:
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    
    # Load model weights
    model.load_weights(model_weights_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Compile model (needed for evaluation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare validation data
# Use the same validation split as training (20%)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

data_dir = os.path.join('data', 'asl_alphabet_train', 'asl_alphabet_train')
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}. Please check the data path.")

validation_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation',  # Use validation subset
    shuffle=False  # Important: don't shuffle for evaluation
)

class_names = list(validation_generator.class_indices.keys())
print(f"\nClass names: {class_names}")
print(f"Number of validation samples: {validation_generator.samples}")
print("=" * 50)

# Evaluate model
print("\nEvaluating model on validation set...")
print("=" * 50)

# Get predictions
print("Generating predictions...")
validation_generator.reset()
predictions = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = validation_generator.classes

# Limit to actual number of samples
num_samples = len(true_classes)
predicted_classes = predicted_classes[:num_samples]
predictions = predictions[:num_samples]

# Calculate overall accuracy
correct_predictions = np.sum(predicted_classes == true_classes)
overall_accuracy = correct_predictions / num_samples * 100

print(f"\n{'='*50}")
print("TEST RESULTS")
print(f"{'='*50}")
print(f"Total validation samples: {num_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print(f"{'='*50}")

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-" * 50)
for i, class_name in enumerate(class_names):
    class_mask = true_classes == i
    if np.sum(class_mask) > 0:
        class_correct = np.sum((predicted_classes == i) & class_mask)
        class_total = np.sum(class_mask)
        class_accuracy = class_correct / class_total * 100
        print(f"{class_name:10s}: {class_correct:3d}/{class_total:3d} = {class_accuracy:6.2f}%")

# Classification report
print("\n" + "=" * 50)
print("Detailed Classification Report:")
print("=" * 50)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion Matrix
print("\n" + "=" * 50)
print("Confusion Matrix:")
print("=" * 50)
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix (rows = true, columns = predicted):")
print(" " * 12, end="")
for name in class_names:
    print(f"{name:>8}", end="")
print()
for i, name in enumerate(class_names):
    print(f"{name:12s}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i][j]:8d}", end="")
    print()

# Calculate per-class metrics
print("\n" + "=" * 50)
print("Per-Class Metrics:")
print("=" * 50)
for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Sample predictions
print("\n" + "=" * 50)
print("Sample Predictions (first 10):")
print("=" * 50)
for i in range(min(10, num_samples)):
    true_class = class_names[true_classes[i]]
    pred_class = class_names[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]] * 100
    status = "CORRECT" if true_classes[i] == predicted_classes[i] else "WRONG"
    print(f"Sample {i+1:3d}: True={true_class:6s} | Predicted={pred_class:6s} | Confidence={confidence:5.2f}% | {status}")

print("\n" + "=" * 50)
print("Testing completed!")
print("=" * 50)

