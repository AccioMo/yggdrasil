import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

california = fetch_california_housing()

print (california.target)

data = pd.DataFrame(california.data, columns=california.feature_names)
target = pd.Series(california.target)

print(data.head())

correlation_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of California Housing Dataset')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

print("Training set size:", train_X.shape)
print("Test set size:", test_X.shape)

tree = RandomForestRegressor()
tree.fit(train_X, train_y)

predictions = tree.predict(test_X)

plt.figure(figsize=(10, 6))
plt.scatter(test_y, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()
