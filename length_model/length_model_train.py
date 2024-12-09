import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Directory paths
train_data_dir = 'length_dataset/train/'
validate_data_dir = 'length_dataset/validate/'

# Function to load and preprocess images
def parse_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [96, 192])  # Set to height 96, width 192
    img = img / 255.0  # Normalize to [0,1] range
    return img

# Function to extract label (captcha length) from filename
def get_label(file_path):
    # Get the file name from the path
    file_name = tf.strings.split(file_path, os.sep)[-1]
    
    # Remove the file extension ".png"
    base_name = tf.strings.regex_replace(file_name, r"\.png$", "")
    
    # Split by underscore and take the first part if an underscore exists
    main_part = tf.strings.split(base_name, '_')[0]
    
    # Calculate the length of the label (main_part)
    label_length = tf.strings.length(main_part) - 1 
    
    # Uncomment to print details for debugging
    # tf.print("File name:", file_name, ", Main part:", main_part, ", Label (length):", label_length)
    
    return label_length

# Load dataset function
def load_dataset(data_dir, batch_size=32):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
    file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)

    images_ds = file_paths_ds.map(parse_image)
    labels_ds = file_paths_ds.map(get_label)
    # Combine images and labels, then shuffle and batch
    dataset = tf.data.Dataset.zip((images_ds, labels_ds))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset

# Load training and validation datasets
train_dataset = load_dataset(train_data_dir)
validate_dataset = load_dataset(validate_data_dir)

# Model definition
model = tf.keras.models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(96, 192, 3)),  # Updated input shape
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  # Predicting lengths 1 to 6
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=3,              # Stop if no improvement after 3 epochs
    restore_best_weights=True  # Restore weights from the best epoch
)

# Train the model with early stopping
model.fit(
    train_dataset,
    validation_data=validate_dataset,
    epochs=20,
    callbacks=[early_stopping]  # Add early stopping callback
)

# Save the model as a .keras file
model.save('captcha_model_disc.keras')
print("Model saved as captcha_model_disc.keras")
