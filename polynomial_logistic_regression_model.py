import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(__file__))
df = pd.read_csv("nonlinear_data.csv")
df = df.dropna()  # dropping small percentage of rows have missing values
X_full = df.iloc[:, 0:2].values  # Features (columns 1-2)
y_full = df['label'].values  # Target variable (column 3)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1  # Example: pos = [False, False, False, True, True, True]
    neg = y == 0  # Example: neg = [True, True, True, False, False, False]
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c="#ea6464", label=pos_label)
    # X[pos, 0] : Take rows where pos=True, Take column 0 (x1)
    # X[pos, 1] : Take rows where pos=True, Take column 1 (x1)

    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label,
               facecolors='none', edgecolors="#65A6DF", lw=2.5)
    # X[neg, 0] : Take rows where neg=True, Take column 0 (x1)
    # X[neg, 1] : Take rows where neg=True, Take column 1 (x1)

    ax.legend(loc=loc)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("poly", PolynomialFeatures(degree=6, include_bias=False)),
    ("logreg", LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Prediction on test set:", y_pred)
print("Accuracy on training set:", model.score(X_train, y_train))
print("Accuracy on test set:", model.score(X_test, y_test))

# Simple decision boundary plot
x_min, x_max = X_full[:, 0].min() - 0.5, X_full[:, 0].max() + 0.5
y_min, y_max = X_full[:, 1].min() - 0.5, X_full[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
plot_data(X_full, y_full, ax, s=70)

ax.contourf(xx, yy, Z, alpha=0.22, levels=2, cmap='coolwarm', extend='both')
ax.contour(xx, yy, Z, levels=[0, 1], colors='black', linewidths=2)
ax.set_xlabel("Feature 1", fontsize=12)
ax.set_ylabel("Feature 2", fontsize=12)
ax.set_title("Polynomial Logistic Regression with Decision Boundary", fontsize=14)
ax.grid(True, alpha=0.25)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='best')
plt.tight_layout()
plt.show()