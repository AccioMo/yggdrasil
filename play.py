import pandas as pd
from collections import Counter

class TreeNode:
	def __init__(self, feature=None, threshold=None):
		self.feature = feature
		self.threshold = threshold
		self.left = None
		self.right = None

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
			unique_thresholds = sorted(X[feature].unique().tolist())
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
	
	def _build_tree(self, X, y):
		"""build decision tree from DataFrame arg"""

		feature, threshold = self._best_split(X, y)
		if (feature is None or threshold is None):
			return None
		node = TreeNode(feature, threshold)
		print(feature, threshold)
		left_indices = X[X[feature] == threshold].index
		right_indices = X[X[feature] != threshold].index
		print(X, y)
		X2 = X.loc[left_indices]
		y2 = y.loc[left_indices]
		X1 = X.loc[right_indices]
		y1 = y.loc[right_indices]
		print(X, y)
		node.left = self._build_tree(X1, y1)
		node.right = self._build_tree(X2, y2)
		return (node)

	def fit(self, X, y):
		"""train the decision tree on the data"""
		self.root = self._build_tree(X, y)


df = pd.read_csv('data.csv')

yggdrasil = Tree()

print(yggdrasil.fit(df.iloc[:, 1:-1], df.iloc[:, -1:]))
