class OnlineMeanCalculator:
    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count

    def get_mean(self):
        return self.mean

# Example usage
mean_calculator = OnlineMeanCalculator()

values = [10, 20, 30, 40, 50]

for value in values:
    mean_calculator.update(value)

final_mean = mean_calculator.get_mean()
print(f"Mean: {final_mean:.2f}")


from collections import deque

class RollingMeanCalculator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.mean = 0.0

    def update(self, value):
        if len(self.values) == self.window_size:
            self.mean -= self.values.popleft() / self.window_size
        self.values.append(value)
        self.mean += value / self.window_size

    def get_mean(self):
        return self.mean

# Example usage
rolling_mean_calculator = RollingMeanCalculator(window_size=3)

values = [10, 20, 30, 40, 50]

for value in values:
    rolling_mean_calculator.update(value)

final_mean = rolling_mean_calculator.get_mean()
print(f"Mean of the last {rolling_mean_calculator.window_size} values: {final_mean:.2f}")

