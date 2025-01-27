import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# יצירת נתונים בצורת מעגל בדו-ממד
np.random.seed(0)
n_samples = 100
radius = np.random.rand(n_samples)
angle = 2 * np.pi * np.random.rand(n_samples)
x1 = radius * np.cos(angle)
x2 = radius * np.sin(angle)

# קביעת התוויות (למשל, נקודות בתוך המעגל לעומת מחוץ למעגל)
y = (radius > 0.5).astype(int)

# הרחבת מימדים - הוספת מימד שלישי
x3 = x1**2 + x2**2  # פונקציה ריבועית שמשלבת את x1 ו-x2
X = np.column_stack((x1, x2, x3))

# יצירת מודל SVM עם גרעין לינארי
model = SVC(kernel="linear")
model.fit(X, y)

# יצירת גרף בדו-ממד
plt.figure(figsize=(8, 8))
plt.scatter(x1[y == 0], x2[y == 0], color='blue', label='Class 0')
plt.scatter(x1[y == 1], x2[y == 1], color='red', label='Class 1')
plt.title("Data in 2D (Circle)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

# גרף בתלת-ממד שמציג את הנתונים לאחר הוספת המימד השלישי
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], color='blue', label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', label='Class 1')
ax.set_title("Data in 3D (Transformed Space)")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
plt.legend()
plt.show()

# תצוגת משטח ההפרדה (ההיפר-מישור)
coef = model.coef_[0]
intercept = model.intercept_
xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], color='blue', label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', label='Class 1')
ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')
ax.set_title("Linear Decision Boundary in 3D")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
plt.legend()
plt.show()
