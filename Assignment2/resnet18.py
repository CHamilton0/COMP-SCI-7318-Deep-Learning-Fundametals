import tensorflow as tf

from keras import datasets, layers, models
import matplotlib.pyplot as plt

class IdentityBlock(tf.keras.Model):
    """
    This class will create an identity block for the resnet model
    by subclassing Model class
    """
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name = "")

        self.conv1 = layers.Conv2D(filters, kernel_size, 
                                   padding = 'same')
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filters, kernel_size, 
                                   padding = 'same')
        self.bn2 = layers.BatchNormalization()
        
        self.act = layers.Activation('relu')
        self.add = layers.Add()
        
        
    def call(self, input_tensor):
        # Block 1: Conv--> BN--> ReLU
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)
        
        # Block 2: Conv--> BN--> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        # skip connection
        x = self.add([x, input_tensor])
        x = self.act(x)
        return x

class ResNet18(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.conv = layers.Conv2D(64, 7, padding="same")
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("relu")
        self.max_pool = layers.MaxPool2D((3, 3))
        self.drop_out = layers.Dropout(0.3)

        # create Identity blocks
        self.id1 = IdentityBlock(64, 3)
        self.id2 = IdentityBlock(64, 3)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)
        x = self.drop_out(x)

        # call identity blocks
        x = self.id1(x)
        x = self.drop_out(x)
        x = self.id2(x)
        x = self.drop_out(x)

        x = self.global_pool(x)
        return self.classifier(x)


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

NUM_CLASSES = 10
resnet18 = ResNet18(NUM_CLASSES)

resnet18.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = resnet18.fit(
    train_images,
    train_labels,
    epochs=10,
    #   batch_size = BATCH_SIZE,
    validation_data=(test_images, test_labels),
)

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")

test_loss, test_acc = resnet18.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
