import numpy as np

from load_diabetes import load_diabetes_data

training, test = load_diabetes_data("diabetes_scale.txt", 8)

N = 0.01
T = 1000
w = np.zeros(training[0][0].shape[0])

for t in range(T):

    train_sum = np.zeros(training[0][0].shape[0])
    for i in range(len(training)):
        y = training[i][1]
        x = training[i][0]
        
        try:
            if y * np.dot(x, w) <= 0:
                train_sum += y * x
        except:
            print(f"Warning failed to train on data {x}")
            pass

    w = w + N * train_sum

correct = 0
total = 0
for data in test:
    label = data[1]
    input = data[0]

    try:
        predicted = np.sign(np.dot(input, w))
    except:
        print(f"Warning failed to predict data {input}")
        continue

    if predicted == label:
        correct += 1
    total += 1

print(w)
print((correct/total) * 100)
