import pickle


def unpickle(file):

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


x = unpickle(
    "C:\\Users\\chris\\OneDrive\\University\\2024\\Deep Learning Fundamentals\\assignment2\\cifar-10-batches-py\\data_batch_1"
)

x = {y.decode("ascii"): x.get(y) for y in x.keys()}

# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
print(x["data"])
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
print(x["labels"])
