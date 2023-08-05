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
