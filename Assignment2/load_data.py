import pickle
import numpy as np


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
        "/home/chris/uni/cifar-10-batches-py/data_batch_5",
    ]

    data = []
    labels = np.array([])
    for path in data_paths:
        batch_data = get_data(path)
        data.append(batch_data["data"])
        labels = np.append(labels, batch_data["labels"])

    data = np.concatenate(data, axis=0)

    # data -- a 50000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 50000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
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
