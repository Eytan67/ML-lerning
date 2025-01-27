import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# יצירת נתונים מדומים
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# חלוקה לנתוני אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# יצירת המודל של רגרסיה לינארית
linear_regressor = LinearRegression()

# אימון המודל
linear_regressor.fit(X_train, y_train)

# חיזוי בעזרת המודל
y_pred = linear_regressor.predict(X_test)

# # הצגת התוצאות


# יצירת המודל של רגרסיה רידג' עם פרמטר אלפא
ridge_regressor = Ridge(alpha=1.0)

# אימון המודל
ridge_regressor.fit(X_train, y_train)

# חיזוי בעזרת המודל
y_pred_ridge = ridge_regressor.predict(X_test)

# הצגת התוצאות


# יצירת המודל של רגרסיה לאסו עם פרמטר אלפא
lasso_regressor = Lasso(alpha=0.1)

# אימון המודל
lasso_regressor.fit(X_train, y_train)

# חיזוי בעזרת המודל
y_pred_lasso = lasso_regressor.predict(X_test)

# הצגת התוצאות



# חישוב הביצועים עבור כל מודל
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Absolute Error (MAE): {mae:.3f}')
    print(f'Mean Squared Error (MSE): {mse:.3f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
    print(f'R^2: {r2:.3f}')
    
# אחרי שתשלים את חיזוי המודל, אתה יכול לקרוא לפונקציה כך:
print("Linear Regression Results:")
evaluate_model(y_test, y_pred)

print("\nRidge Regression Results:")
evaluate_model(y_test, y_pred_ridge)

print("\nLasso Regression Results:")
evaluate_model(y_test, y_pred_lasso)




# ביצוע אימון עם ערכים שונים של alpha
alphas = [0.1, 1, 10, 100]

# Ridge Regression
for alpha in alphas:
    ridge_regressor = Ridge(alpha=alpha)
    ridge_regressor.fit(X_train, y_train)
    y_pred_ridge = ridge_regressor.predict(X_test)
    print(f"Ridge Regression (alpha={alpha}):")
    evaluate_model(y_test, y_pred_ridge)
    print("-" * 40)

# Lasso Regression
for alpha in alphas:
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train, y_train)
    y_pred_lasso = lasso_regressor.predict(X_test)
    print(f"Lasso Regression (alpha={alpha}):")
    evaluate_model(y_test, y_pred_lasso)
    print("-" * 40)
