import numpy as np
import matplotlib.pyplot as plt

# Original sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Stretched sigmoid function with scaling factor 'a' and translation factor 'b'
def stretched_sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

# Generate x values from -10 to 10
x = np.linspace(-70, 70, 1000)

# Original sigmoid values
y_sigmoid = sigmoid(x)

# Stretched sigmoid values with a=2 and b=0
y_stretched = stretched_sigmoid(x, a=0.08, b=0)

# Plot the sigmoid functions
plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label="Original Sigmoid")
plt.plot(x, y_stretched, label="Stretched Sigmoid (a=0.08, b=0)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Original Sigmoid vs. Stretched Sigmoid")
plt.legend()
plt.grid(True)
plt.show()
