
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	y_mean = np.mean(y_true)
	sst = np.sum((y_true - y_mean)**2)
	ssr = np.sum((y_true - y_pred)**2)
	return 1 - ssr/sst