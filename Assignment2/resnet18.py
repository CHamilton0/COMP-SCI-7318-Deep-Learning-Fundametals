import tensorflow as tf

import keras
from keras import layers


class IdentityBlock(tf.keras.Model):
    """
    This class will create an identity block for the resnet model
    by subclassing Model class
    """

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name="")

        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same")
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.act = layers.Activation("relu")
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


def compile_resnet_model(num_classes, learning_rate):
    resnet18 = ResNet18(num_classes)

    resnet18.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return resnet18
