import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def plot_predictions_vs_actual(test_y, predictions_a, predictions_b, predictions_c):
    plt.figure(figsize=(10, 6))
    plt.scatter(test_y, predictions_a, color='red', label='Linear Regression', alpha=0.5)
    plt.scatter(test_y, predictions_b, color='green', label='Decision Tree', alpha=0.5)
    plt.scatter(test_y, predictions_c, color='orange', label='Random Forest', alpha=0.5)
    plt.legend()
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2)
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.close()

california = fetch_california_housing()

data = pd.DataFrame(california.data, columns=california.feature_names)
target = pd.Series(california.target)

train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_X, train_y)

predictions_a = model.predict(test_X)

tree = DecisionTreeRegressor(criterion='friedman_mse', random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
tree.fit(train_X, train_y)

predictions_b = tree.predict(test_X)

random_forest = RandomForestRegressor()
random_forest.fit(train_X, train_y)

predictions_c = random_forest.predict(test_X)

plot_predictions_vs_actual(test_y, predictions_a, predictions_b, predictions_c)
