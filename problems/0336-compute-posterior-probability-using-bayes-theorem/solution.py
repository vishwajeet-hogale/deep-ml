import numpy as np
def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	priors = np.asarray(priors)
	likelihoods = np.asarray(likelihoods)
	res = priors * likelihoods / np.sum(priors * likelihoods) + 1e-7
	return res.tolist()