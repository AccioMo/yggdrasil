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
		pi = list(map(lambda x: x/total, Counter(y[y.columns[0]]).values()))

		impurity = 1.0 - sum(list(map(lambda x: x ** 2, pi)))

		return (impurity)

	def _information_gain(self, y, left_y, right_y):
		"""calculate information gain from a split"""

		parent_impurity = self._gini_impurity(y)

		left_y_gini = self._gini_impurity(left_y)
		right_y_gini = self._gini_impurity(right_y)
		child_impurity = left_y_gini*len(left_y)/len(y) + right_y_gini*len(right_y)/len(y)

		return (parent_impurity - child_impurity)

	def _best_split(self, X, y):
		"""find the best split for the tree"""

		highest_gain = 0
		best_threshold = None
		best_feature = None
		# unique_thresholds = sorted(set(X.values.flatten()))

		for feature in X.columns:
			unique_thresholds = X[feature].unique().tolist()
			for threshold in unique_thresholds:
				left_child = y.loc[X[X[feature] == threshold].index]
				right_child = y.loc[X[X[feature] != threshold].index]
				gain = self._information_gain(y, left_child, right_child)
				print(f"{gain:.3f} ({feature} = {threshold})")
				if (gain > highest_gain):
					highest_gain = gain
					best_threshold = threshold
					best_feature = feature
		return (best_feature, best_threshold)
	
	def build_tree(self, X, y):
		"""build decision tree from DataFrame arg"""

		return (self._best_split(X, y))


df = pd.read_csv('data.csv')

yggdrasil = Tree()

print(yggdrasil.build_tree(df.iloc[:, 1:-1], df.iloc[:, -1:]))
