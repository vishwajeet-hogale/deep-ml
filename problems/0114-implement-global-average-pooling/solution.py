import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	return np.mean(x, axis= (0,1)).flatten()