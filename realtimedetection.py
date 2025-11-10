import cv2
import numpy as np
import json
import os

# Check if model files exist
model_keras_path = "signlanguagedetectionmodel48x48.keras"
model_json_path = "signlanguagedetectionmodel48x48.json"
model_weights_path = "signlanguagedetectionmodel48x48.h5"

if not os.path.exists(model_keras_path) and not os.path.exists(model_json_path):
    raise FileNotFoundError(f"Model files not found. Please train the model first.\n"
                          f"Expected files: {model_keras_path} or {model_json_path}")

# Try different import methods for TensorFlow/Keras
model = None
try:
    # Try using keras standalone first (Keras 3)
    from keras.models import load_model # type: ignore
    print("Loading model using Keras standalone...")
    if os.path.exists(model_keras_path):
        model = load_model(model_keras_path)
        print("Model loaded successfully using .keras file!")
except ImportError:
    try:
        # Fallback to tensorflow.keras
        from tensorflow.keras.models import load_model # type: ignore
        print("Loading model using TensorFlow Keras...")
        if os.path.exists(model_keras_path):
            model = load_model(model_keras_path)
            print("Model loaded successfully using .keras file!")
    except (ImportError, Exception) as e:
        # Last resort: load from JSON + H5
        if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
            from tensorflow.keras.models import model_from_json # type: ignore
            print("Loading model from JSON and H5 files...")
            try:
                with open(model_json_path, "r") as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json)
                model.load_weights(model_weights_path)
                print("Model loaded successfully!")
            except Exception as e2:
                print(f"Error loading model from JSON/H5: {e2}")
                raise
        else:
            raise FileNotFoundError(f"Model files not found: {model_json_path} or {model_weights_path}")

if model is None:
    raise RuntimeError("Failed to load model. Please check that model files exist and are valid.")

# Load class names from file if it exists, otherwise use default alphabetical order
class_names_path = "class_names.json"
if os.path.exists(class_names_path):
    try:
        with open(class_names_path, "r") as f:
            label = json.load(f)
        print(f"Loaded {len(label)} classes from class_names.json")
    except Exception as e:
        print(f"Warning: Could not load class_names.json: {e}")
        print("Using default class order...")
        # Default alphabetical order for 29 classes (A-Z, del, nothing, space)
        label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                 'del', 'nothing', 'space']
        print(f"Using default class order with {len(label)} classes")
else:
    # Default alphabetical order for 29 classes (A-Z, del, nothing, space)
    label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
             'del', 'nothing', 'space']
    print(f"Using default class order with {len(label)} classes")
    print("Note: Run train_model.py first to generate class_names.json for correct class mapping")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Please check if your camera is connected.")
    exit()

print("\n" + "="*50)
print("Sign Language Detection - Real-time")
print("="*50)
print("Instructions:")
print("- Position your hand in the orange rectangle (top-left)")
print(f"- Show sign language gestures: {', '.join(label)}")
print("- Press 'q' to quit")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break
    
    # Draw rectangle for hand detection area
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 2)
    
    # Extract and preprocess the hand region
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (48, 48))
    cropframe = extract_features(cropframe)
    
    # Make prediction
    pred = model.predict(cropframe, verbose=0)
    prediction_label = label[pred.argmax()]
    confidence = np.max(pred) * 100
    
    # Draw prediction area
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    
    # Display prediction
    if prediction_label in ['nothing', 'space']:
        # For 'nothing' or 'space', show a space or the label
        cv2.putText(frame, prediction_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(confidence)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow("Sign Language Detection - Press 'q' to quit", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nExiting...")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")