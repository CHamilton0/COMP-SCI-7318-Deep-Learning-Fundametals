import tensorflow as tf
import keras
import numpy as np
from load_data import get_training_data, get_test_data
import keras

# Load your dataset
training_data = get_training_data()
test_data = get_test_data()

# Define the number of classes and image size
num_classes = 10
image_size = 224  # VGG19 input requirement

# Limit GPU memory growth (optional, helps manage GPU memory more effectively)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Create a generator function to preprocess images on the fly
def data_generator(data, labels, batch_size=8):
    num_samples = len(data)
    
    while True:  # Loop forever for Keras' fit method
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_images = []
            batch_labels = []
            
            for i in batch_indices:
                img = data[i]
                red = img[:1024].reshape((32, 32))
                green = img[1024:2048].reshape((32, 32))
                blue = img[2048:].reshape((32, 32))
                img_rgb = np.stack((red, green, blue), axis=-1)
                
                # Resize and preprocess for VGG19
                img_resized = tf.image.resize(img_rgb, (image_size, image_size)).numpy()
                img_preprocessed = tf.keras.applications.vgg19.preprocess_input(img_resized)
                
                batch_images.append(img_preprocessed)
                batch_labels.append(tf.keras.utils.to_categorical(labels[i], num_classes))
            
            yield np.array(batch_images), np.array(batch_labels)

# Create training and validation generators
train_generator = data_generator(training_data["data"], training_data["labels"], batch_size=8)
validation_generator = data_generator(test_data["data"], test_data["labels"], batch_size=8)

# Define and compile the model
mobilenet = keras.applications.MobileNet(include_top=False, weights=None, input_shape=(224, 224, 3))

my_new_model = keras.models.Sequential([
    mobilenet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(num_classes, activation='softmax')
])

my_new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model using the generator
steps_per_epoch = len(training_data["data"]) // 64
validation_steps = len(test_data["data"]) // 64

my_new_model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10, validation_data=validation_generator, validation_steps=validation_steps)
