import pandas as pd
from collections import Counter

class Tree:
	def __init__(self, df=None):
		self.root = None
		self.df = df

	def _gini_impurity(self, y):
		"""calculate gini impurity"""
		if (y.empty):
			return (0)

		total = len(y)
		pi = list(map(lambda x: x/total, Counter(y).values()))

		racism_value = 1 - sum(list(map(lambda x: x ** 2, pi)))

		return (racism_value)
	
	def build_tree(self, X, y):
		"""build decision tree from DataFrame arg"""
		for column in X.columns:
			unique_values = X[column].unique().tolist()
			for value in unique_values:
				right_child = X[X[column] == value]['Play_Tennis']
				right_child_gini = self._gini_impurity(right_child)
				print(column, "==", value, ": ", right_child_gini)
				left_child = X[X[column] != value]['Play_Tennis']
				left_child_gini = self._gini_impurity(left_child)
				print(column, "=/=", value, ": ", left_child_gini)
				weighted_child_gini = right_child_gini*len(right_child)/len(X) + left_child_gini*len(left_child)/len(X)
				print(f"{right_child_gini}x{len(right_child)}/{len(X)} + {left_child_gini}x{len(left_child)}/{len(X)} = {weighted_child_gini}")
				print("wcg:", weighted_child_gini)


df = pd.read_csv('data.csv')

yggdrasil = Tree()

print(yggdrasil.build_tree(df.iloc[:, 1:], df.iloc[:, :]))
