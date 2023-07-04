import numpy as np
def control_to_onehot(control_values, num_bins = 15):
    # Convert control values to one-hot array
    control_value_x = control_values[0]
    control_value_y = control_values[1]

    index_x = int(((control_value_x + 1) / 2) * (num_bins - 1))
    index_y = int(((control_value_y + 1) / 2) * (num_bins - 1))

    index = index_y * num_bins + index_x

    onehot = np.zeros(num_bins * num_bins)
    onehot[index] = 1

    return onehot

def onehot_to_control(onehot, num_bins = 15):
    # Convert one-hot array to control values
    index = np.argmax(onehot)

    index_x = index % num_bins
    index_y = index // num_bins

    control_value_x = (index_x / (num_bins - 1) * 2) - 1
    control_value_y = (index_y / (num_bins - 1) * 2) - 1

    return np.array([control_value_x, control_value_y])


one_hot =control_to_onehot(control_values = [0.1, 1])
print(one_hot)

print(onehot_to_control(one_hot))