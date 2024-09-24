import numpy as np
import random

def load_diabetes_data(file_name, num_vars):
    training_data = []
    test_data = []

    all_data = []

    with open(file_name) as diabetes_file:
        for line in diabetes_file:
            values = line.split()
            label = int(values[0])

            variables = [0] * num_vars

            for value in values[1:]:
                value = value.replace("\n", "")
                index = int(value.split(":")[0])
                value = float(value.split(":")[1])
                variables[index - 1] = value

            all_data.append((np.array(variables), label))

        random.shuffle(all_data)

        training_data = all_data[:int((len(all_data)+1)*.80)]
        test_data = all_data[int((len(all_data)+1)*.80):]

    return training_data, test_data
