import tensorflow as tf
import matplotlib.pyplot as plt

from load_data import get_training_data, get_test_data, data_generator
from mobilenet import compile_mobilenet_model
from mobilenetv2 import compile_mobilenet_v2_model
from mobilenetv3 import compile_mobilenet_v3_model
from alexnet import compile_alexnet_model
from resnet18 import compile_resnet_model

# Limit GPU memory growth (optional, helps manage GPU memory more effectively)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load your dataset
training_data = get_training_data()
test_data = get_test_data()

# Define the number of classes and image size
num_classes = 10
image_size = 32

# Create training and validation generators
train_generator = data_generator(
    training_data["data"], training_data["labels"], image_size, image_size, batch_size=8
)
validation_generator = data_generator(
    test_data["data"], test_data["labels"], image_size, image_size, batch_size=8
)

# Fit the model using the generator
steps_per_epoch = len(training_data["data"]) // 16
validation_steps = len(test_data["data"]) // 16

mobilenet_model = compile_mobilenet_model(image_size, image_size, num_classes)
mobilenet_v2_model = compile_mobilenet_v2_model(image_size, image_size, num_classes)
mobilenet_v3_model = compile_mobilenet_v3_model(image_size, image_size, num_classes)
alexnet_model = compile_alexnet_model(image_size, image_size, num_classes)
resnet_model = compile_resnet_model(num_classes)

model = resnet_model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)

plt.figure(figsize=(10, 10))
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()
