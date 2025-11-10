import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
import os
import shutil
import json

# Dataset exists locally at data/asl_alphabet_train/asl_alphabet_train/
print("Using local dataset at data/asl_alphabet_train/asl_alphabet_train/")

# Configure GPU if available
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("Using GPU for training")
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("Using CPU for training")

# Data generators with validation split
# validation_split=0.2 means 20% of data will be used for validation
validation_split = 0.2

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)

batch_size = 128

# Use local paths (Windows compatible)
data_dir = os.path.join(os.getcwd(), 'data', 'asl_alphabet_train', 'asl_alphabet_train')

print(f"Data directory: {data_dir}")

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training',  # This is the training subset
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation',  # This is the validation subset
    shuffle=True
)

class_names = list(train_generator.class_indices.keys())
print(f"Class names: {class_names}")

# Build model
model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer - 29 classes (A-Z, del, nothing, space)
model.add(Dense(29, activation='softmax'))

model.summary()

# Compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Setup TensorBoard
logdir = os.path.join("Logs")
if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.makedirs(logdir, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=logdir)

print("\nStarting training...")
print("=" * 50)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[tensorboard_callback]
)

print("\nTraining completed!")
print("=" * 50)

# Save model locally
model_json = model.to_json()
with open("signlanguagedetectionmodel48x48.json",'w') as json_file:
    json_file.write(model_json)
# Save in both H5 and Keras format
model.save("signlanguagedetectionmodel48x48.h5")
model.save("signlanguagedetectionmodel48x48.keras")

# Save class names to a file for later use
class_names_dict = train_generator.class_indices
# Reverse the dictionary to get class_index -> class_name mapping
class_names_reversed = {v: k for k, v in class_names_dict.items()}
# Create sorted list by index
class_names_sorted = [class_names_reversed[i] for i in range(len(class_names_reversed))]
with open("class_names.json", 'w') as f:
    json.dump(class_names_sorted, f, indent=2)
print("Model saved successfully!")
print("Files saved:")
print("  - signlanguagedetectionmodel48x48.json")
print("  - signlanguagedetectionmodel48x48.h5")
print("  - signlanguagedetectionmodel48x48.keras")
print("  - class_names.json")

