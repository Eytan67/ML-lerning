import numpy as np


def predict(x:np.array, w:np.array) -> float:
    return x.dot(w)

def loss_function(X:np.array, w:np.array, Y:float) -> float:
    print(f" X: {X} \n w: {w} \n Y: {Y}")
    return sum((y_i - predict(x_i, w))**2 for x_i, y_i in zip(X, Y))

import numpy as np

def ordinary_least_squares(X, Y):
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    w = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ Y
    return w

def gradient_descent(X, Y, learning_rate=0.01, iterations=1):

    n, p = X.shape

    X_with_bias = np.hstack((np.ones((n, 1)), X))

    # אתחול המשקלות באפסים
    w = np.zeros(p+1)

    for i in range(iterations):
        error = loss_function(X_with_bias, w, Y)
        print(f"error: {error}")

        # חישוב הגרדיאנט
        gradient = (2 / n) * (X_with_bias.T @ error)

        # עדכון המשקלות
        w -= learning_rate * gradient

        # הדפסת פונקציית האובדן כל 100 איטרציות (אופציונלי)
        if i % 100 == 0:
            loss = np.mean(error ** 2)
            # print(f"Iteration {i}: Loss = {loss:.4f}")

    return w

# נתונים לדוגמה
X = np.array([[1, 2], [2, 3], [3, 4]])  # מטריצת פיצ'רים (n x p)
Y = np.array([5, 7, 9])  # וקטור תוויות (n x 1)

# חישוב המקדמים
# w = ordinary_least_squares(X, Y)

# print("המקדמים (כולל האיבר החופשי):", w)


w = gradient_descent(X, Y, 0.01, 1000)
print("המקדמים (כולל האיבר החופשי):", w)
