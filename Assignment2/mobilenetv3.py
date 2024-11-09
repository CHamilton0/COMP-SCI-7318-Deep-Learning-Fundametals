import keras
import keras

def compile_mobilenet_v3_model(image_width, image_height, num_classes, learning_rate):
    # Define and compile the model
    mobilenet = keras.applications.MobileNetV3Large(
        include_top=False, weights=None, input_shape=(image_width, image_height, 3)
    )

    mobilenet_model = keras.models.Sequential(
        [
            mobilenet,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    mobilenet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return mobilenet_model
