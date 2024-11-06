import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.image import resize

# Define model parameters
image_size = 224
num_classes = 10

# Example function to load your data (replace with your actual loading method)
def load_data():
    # Assuming `get_training_data` and `get_test_data` load your data in a dictionary
    from load_data import get_training_data, get_test_data
    training_data = get_training_data()
    test_data = get_test_data()
    return training_data, test_data

# Load your data
training_data, test_data = load_data()

print("Training data shape:", training_data["data"].shape)
print("Training labels shape:", training_data["labels"].shape)
print("Test data shape:", test_data["data"].shape)
print("Test labels shape:", test_data["labels"].shape)

# Define the preprocessing function
def preprocess_image(image, label):
    # Verify the image length and cast it to float32 for compatibility with VGG16
    if tf.shape(image)[0] < 3072:
        print("Skipping image due to incorrect shape:", tf.shape(image))
        return None, None

    # Reshape to separate color channels
    try:
        red = tf.reshape(image[:1024], (32, 32))
        green = tf.reshape(image[1024:2048], (32, 32))
        blue = tf.reshape(image[2048:], (32, 32))
        img_rgb = tf.stack((red, green, blue), axis=-1)

        # Resize and apply VGG16 preprocessing
        img_resized = resize(img_rgb, (image_size, image_size))
        img_resized = tf.cast(img_resized, tf.float32)
        img_preprocessed = preprocess_input(img_resized)

        return img_preprocessed, tf.keras.utils.to_categorical(label, num_classes=num_classes)
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None, None

# Create TensorFlow Dataset
def create_tf_dataset(data, labels, batch_size=32, shuffle_buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Filter out images not having the expected length of 3072
    dataset = dataset.filter(lambda img, lbl: tf.shape(img)[0] == 3072)

    # Apply preprocessing
    dataset = dataset.map(lambda img, lbl: tf.py_function(
        preprocess_image, [img, lbl], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Remove any samples where preprocessing returned None
    dataset = dataset.filter(lambda img, lbl: img is not None and lbl is not None)

    # Set explicit shapes
    dataset = dataset.map(lambda img, lbl: (tf.ensure_shape(img, [image_size, image_size, 3]),
                                            tf.ensure_shape(lbl, [num_classes])),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Optimize data loading
    return dataset

# Create datasets
batch_size = 32
train_dataset = create_tf_dataset(training_data["data"], training_data["labels"], batch_size=batch_size)
test_dataset = create_tf_dataset(test_data["data"], test_data["labels"], batch_size=batch_size)

# Define the model
my_new_model = Sequential([
    VGG16(include_top=False, pooling='avg', weights='imagenet'),
    Dense(num_classes, activation='softmax')
])

# Freeze the VGG16 base model
my_new_model.layers[0].trainable = False

# Compile the model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
my_new_model.fit(train_dataset, epochs=10, validation_data=test_dataset)
