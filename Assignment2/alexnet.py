import tensorflow as tf

import keras

def compile_alexnet_model(image_width, image_height, num_classes, learning_rate):
    alexnet = tf.keras.models.Sequential(
        [
            tf.keras.layers.Resizing(image_width, image_height, input_shape=(32, 32, 3)),
            # 1st conv
            tf.keras.layers.Conv2D(
                96,
                (11, 11),
                strides=(4, 4),
                activation="relu",
                input_shape=(image_width, image_height, 3),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
            # 2nd conv
            tf.keras.layers.Conv2D(
                256, (11, 11), strides=(1, 1), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            # 3rd conv
            tf.keras.layers.Conv2D(
                384, (3, 3), strides=(1, 1), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            # 4th conv
            tf.keras.layers.Conv2D(
                384, (3, 3), strides=(1, 1), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            # 5th Conv
            tf.keras.layers.Conv2D(
                256, (3, 3), strides=(1, 1), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
            # To Flatten layer
            tf.keras.layers.Flatten(),
            # To FC layer 1
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.8),
            # To FC layer 2
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    alexnet.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return alexnet
