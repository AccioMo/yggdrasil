import pandas as pd
from collections import Counter

class TreeNode:
	def __init__(self, feature=None, threshold=None, decision=None):
		self.feature = feature
		self.threshold = threshold
		self.left = None
		self.right = None
		self.decision = decision

class Tree:
	def __init__(self, df=None, max_depth=5, min_samples_split=2):
		self.root = None
		self.df = df
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split

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
				left_child = y.loc[X[X[feature] <= threshold].index]
				right_child = y.loc[X[X[feature] > threshold].index]
				gain = self._information_gain(y, left_child, right_child)
				if (gain > highest_gain):
					highest_gain = gain
					best_threshold = threshold
					best_feature = feature
		return (best_feature, best_threshold)
	
	def _build_tree(self, X, y, depth=0):
		"""build decision tree from DataFrame arg"""

		if (y.empty):
			return (TreeNode(decision=None))
		
		decision = y.mode().iloc[0, 0]
		if (Counter(y[y.columns[0]]).most_common(1)[0][1] == len(y)):	### if pure node ###
			print(f"Pure node found at depth {depth} with decision: {decision}")
			return (TreeNode(decision=decision))
		elif (depth >= self.max_depth):									### if max depth reached ###
			print(f"Max depth reached at depth {depth} with decision: {decision}")
			return (TreeNode(decision=decision))
		elif (len(X) <= self.min_samples_split):						### if not enough data ###
			print(f"Minimum samples split reached at depth {depth} with decision: {decision}")
			return (TreeNode(decision=decision))

		feature, threshold = self._best_split(X, y)

		if (feature is None or threshold is None):
			print(f"No valid split found at depth {depth} with decision: {decision}")
			return (TreeNode(decision=decision))
		
		node = TreeNode(feature=feature, threshold=threshold)

		left_indices = X[X[feature] <= threshold].index
		right_indices = X[X[feature] > threshold].index
		X1 = X.loc[left_indices]
		y1 = y.loc[left_indices]
		X2 = X.loc[right_indices]
		y2 = y.loc[right_indices]

		node.left = self._build_tree(X1, y1, depth + 1)
		node.right = self._build_tree(X2, y2, depth + 1)
		return (node)

	def fit(self, X, y):
		"""train the decision tree on the data"""
		self.root = self._build_tree(X, y)


df = pd.read_csv('Housing_Data.csv')

yggdrasil = Tree()

yggdrasil.fit(df.iloc[:, 1:-1], df.iloc[:, -1:])

def print_tree(node):
	"""tree printer using the rich library for better formatting"""

	from rich.tree import Tree
	from rich.console import Console
	from rich.text import Text
	
	console = Console()
	
	def build_rich_tree(node, tree_node=None):
		if node is None:
			return
		
		if hasattr(node, 'decision') and node.decision is not None:
			label = Text(f"🍃 Decision: {node.decision}", style="green bold")
		elif hasattr(node, 'feature') and hasattr(node, 'threshold'):
			label = Text(f"🌿 {node.feature} ≤ {node.threshold}", style="blue")
		else:
			label = Text(f"🔵 {str(node)}", style="white")
		
		if tree_node is None:
			tree_node = Tree(label)
			root_tree = tree_node
		else:
			tree_node = tree_node.add(label)
		
		if hasattr(node, 'left') and node.left is not None:
			left_branch = tree_node.add(Text("≤ (True)", style="dim"))
			build_rich_tree(node.left, left_branch)
		
		if hasattr(node, 'right') and node.right is not None:
			right_branch = tree_node.add(Text("> (False)", style="dim"))
			build_rich_tree(node.right, right_branch)
		
		return tree_node if 'root_tree' not in locals() else root_tree
	
	rich_tree = build_rich_tree(node)
	console.print(rich_tree)

print_tree(yggdrasil.root)
