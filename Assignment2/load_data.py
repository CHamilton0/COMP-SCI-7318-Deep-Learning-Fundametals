import pickle
import numpy as np
import tensorflow as tf

num_classes = 10


def unpickle(file):

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_training_data():
    data_paths = [
        "/home/chris/uni/cifar-10-batches-py/data_batch_1",
        "/home/chris/uni/cifar-10-batches-py/data_batch_2",
        "/home/chris/uni/cifar-10-batches-py/data_batch_3",
        "/home/chris/uni/cifar-10-batches-py/data_batch_4",
    ]

    data = []
    labels = np.array([])
    for path in data_paths:
        batch_data = get_data(path)
        data.append(batch_data["data"])
        labels = np.append(labels, batch_data["labels"])

    data = np.concatenate(data, axis=0)

    # data -- a 40000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 40000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    return {"data": data, "labels": labels}


def get_validation_data():
    path = "/home/chris/uni/cifar-10-batches-py/data_batch_5"

    batch_data = get_data(path)
    data = batch_data["data"]
    labels = batch_data["labels"]

    # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    return {"data": data, "labels": labels}


def get_test_data():
    path = "/home/chris/uni/cifar-10-batches-py/test_batch"

    batch_data = get_data(path)
    data = batch_data["data"]
    labels = batch_data["labels"]

    # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    return {"data": data, "labels": labels}


def get_data(path):
    unpickled_data = unpickle(path)

    dataset = {
        key.decode("ascii"): unpickled_data.get(key) for key in unpickled_data.keys()
    }
    data = np.array(dataset["data"])
    labels = np.array(dataset["labels"])

    return {"data": data, "labels": labels}


def data_generator(data, labels, image_width, image_height, batch_size=8):
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
                img_resized = tf.image.resize(
                    img_rgb, (image_width, image_height)
                ).numpy()
                img_preprocessed = tf.keras.applications.vgg19.preprocess_input(
                    img_resized
                )

                batch_images.append(img_preprocessed)
                batch_labels.append(
                    tf.keras.utils.to_categorical(labels[i], num_classes)
                )

            yield np.array(batch_images), np.array(batch_labels)


def test_data_generator(data, labels, image_width, image_height, batch_size=8):
    num_samples = len(data)
    indices = np.arange(num_samples)  # No need to shuffle for evaluation

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

            # Resize and preprocess
            img_resized = tf.image.resize(img_rgb, (image_width, image_height)).numpy()
            img_preprocessed = tf.keras.applications.vgg19.preprocess_input(img_resized)

            batch_images.append(img_preprocessed)

            batch_labels.append(tf.keras.utils.to_categorical(labels[i], num_classes))

        yield np.array(batch_images), np.array(batch_labels)
