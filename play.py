import pandas as pd
from collections import Counter

class Tree:
	def __init__(self, df=None):
		self.root = None
		self.df = df

	def _gini_impurity(self, discimination_label):
		"""calculate gini impurity"""
		if (self.df.empty):
			return (0)

		total = len(df[discimination_label])
		pi = list(map(lambda x: x/total, Counter(df[discimination_label]).values()))

		racism_value = 1 - sum(list(map(lambda x: x ** 2, pi)))

		return (racism_value)
	
	def plant_tree(self):
		"""build decision tree from DataFrame arg"""
		return (self._gini_impurity('Play_Tennis'))


df = pd.read_csv('data.csv')

yggdrasil = Tree(df)

print(yggdrasil.plant_tree())
