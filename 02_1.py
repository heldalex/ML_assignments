import numpy as np

# exercise 1.1.
data = np.array([
    [1, 50.73, 523902.67],
    [42, 41.83, 325104.45],
    [13, 46.54, 434919.86],
    [25, 58.27, 575719.18],
    [63, 72.53, 629274.54],
    [15, 51.47, 390576.98]
])

X = data[:, :2]  # age and area
y = data[:, 2]   # price

# w = (X^T X)^-1 X^T y = [w_age, w_area]
w = np.linalg.inv(X.T @ X) @ X.T @ y

print(w)

# exercise 1.2.
house = np.array([10, 50])
predicted_cost = w @ house

print(predicted_cost)

# exercise 1.3.
real_cost = 427451.1
least_squares_loss = (predicted_cost - real_cost) ** 2
L_1 = abs(real_cost - predicted_cost)

print(least_squares_loss, L_1)