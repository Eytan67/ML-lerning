import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt



def predict(x:np.array, w:np.array) -> float:
    n, p = x.shape
    X_with_bias = np.hstack((np.ones((n, 1)), x))
    return X_with_bias @ w

def loss_function(y_predictions: np.array, y: np.array) -> float:
    # print(f" X: {X} \n w: {w} \n Y: {Y}")
    # return sum((y_i - predict(x_i, w))**2 for x_i, y_i in zip(X, Y))
    return sum((y_i - yp_i)**2 for y_i, yp_i in zip(y, y_predictions))

# def ordinary_least_squares(X, Y):
#     X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
#     w = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ Y
#     return w

def calculate_gradient(n: int, X: np.array, error: np.array) -> np.array:
    gradient = (2 / n) * (X.T @ error)
    return gradient

def batch_gradient_descent(X, Y, learning_rate=0.01, iterations=1, batch_size=33):
    n, p = X.shape
    X_with_bias = np.hstack((np.ones((n, 1)), X))
    w = np.zeros(p+1)
    
    for i in range(iterations):
        indexes = np.random.choice(n, batch_size, replace=False)
        X_batch = X_with_bias[indexes]
        Y_batch = Y.iloc[indexes]
        error = X_batch @ w - Y_batch
        gradient = (2 / batch_size) * (X_batch.T @ error)
        w -= learning_rate * gradient
    return w

def stochastic_gradient_descent_SGD(X, Y, learning_rate=0.01, iterations=1):
    batch_size = 1
    return batch_gradient_descent(X, Y, learning_rate, iterations, batch_size)

def gradient_descent(X, Y, learning_rate=0.01, iterations=1):
    n, p = X.shape
    X_with_bias = np.hstack((np.ones((n, 1)), X))

    # אתחול המשקלות באפסים
    w = np.zeros(p+1)
    loss1 = []
    for i in range(iterations):
        # חישוב התחזיות y_hat
        y_predictions = X_with_bias @ w
        # חישוב השגיאות
        error = y_predictions - Y

        loss1.append(loss_function(y_predictions, Y))

        # חישוב הגרדיאנט
        gradient = calculate_gradient(n, X_with_bias, error)

        # עדכון המשקלות
        w -= learning_rate * gradient

        # הדפסת פונקציית האובדן כל 100 איטרציות (אופציונלי)
        if i % 100 == 0:
            loss = np.mean(error ** 2)
            # print(f"Iteration {i}: Loss = {loss:.4f}")
    plt.plot(loss1, marker='o', linestyle='-', color='b', label='Loss per iteration')
    plt.xlabel('Iteration')  # כיתוב לציר ה-X
    plt.ylabel('Loss')       # כיתוב לציר ה-Y
    plt.title('Loss Function Over Iterations')  # כותרת הגרף
    plt.legend()             # הצגת מקרא (Legend)
    plt.grid(True)           # הצגת גריד ברקע
    plt.show()               # הצגת הגרף
    return w

# נתונים לדוגמה
num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
num2 = [[_] for _ in num_friends_good]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

file_path = "./Advertising.csv"

df = pd.read_csv(file_path)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

# החזרת הנתונים כ-DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)


independent = normalized_df[['Digital', 'Radio', 'Newspaper']]
dependent = normalized_df['Sales']
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=42)

X = np.array(num2)  # מטריצת פיצ'רים (n x p)
Y = np.array(daily_minutes_good)  # וקטור תוויות (n x 1)

# w = gradient_descent(X, Y, 0.01, 1000)
# w = stochastic_gradient_descent_SGD(X, Y, 0.001, 1000)
# w = batch_gradient_descent(X, Y, 0.01, 1000000)
# print("המקדמים (כולל האיבר החופשי):", w)

w = gradient_descent(X_train, y_train, 0.5, 10)
y_predictions = predict(X_test, w)
mae = mean_absolute_error(y_test, y_predictions)
mse = mean_squared_error(y_test, y_predictions)
print(f"Mean Squared Error: {mse:.5f}")
print(f"Mean Absolute Error: {mae:.5f}")
plt.scatter(y_test, y_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()

# w1 = stochastic_gradient_descent_SGD(X_train, y_train, 0.01, 1000)
# print(w1)

# w2 = stochastic_gradient_descent_SGD(X_train, y_train, 0.01, 1000)
# y_predictions = predict(X_test, w2)
# mae = mean_absolute_error(y_test, y_predictions)
# mse = mean_squared_error(y_test, y_predictions)
# print(f"Mean Squared Error: {mse:.5f}")
# print(f"Mean Absolute Error: {mae:.5f}")
# plt.scatter(y_test, y_predictions, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted")
# plt.show()



