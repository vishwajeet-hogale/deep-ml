
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	data_tuples = [tuple(x) for x in data]
	data_dict = Counter(data_tuples)
	return [[data_dict[(1,1)], data_dict[(1, 0)]], [data_dict[(0,1)], data_dict[(0,0)]]]
