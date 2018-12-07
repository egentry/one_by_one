import numpy as np

def split_indices(indices, num_testing=25000, frac_training=.8, seed=0):
	"""reproducibly split indices into training, validation, testing sets

	furthermore, the original order of `indices` doesn't matter

	Assumes a breakdown analogous to the one used by Wu and Boada (2018)
	(arXiv:1810.12913)

	First takes `num_testing` indices for a final testing set.
	Then splits the remaining one into frac_training and 1-frac_training
		among the training and validation sets respectively.
	"""

	shuffled_indices = indices.copy()
	shuffled_indices = np.array(sorted(set(shuffled_indices))) # ensure permutation invariance

	np.random.seed(seed)
	np.random.shuffle(shuffled_indices)

	testing_set = shuffled_indices[:num_testing]

	shuffled_indices = shuffled_indices[num_testing:]
	num_indices_left = shuffled_indices.size
	num_training = int(frac_training * num_indices_left)

	training_set = shuffled_indices[:num_training]
	validation_set = shuffled_indices[num_training:]

	return training_set, validation_set, testing_set
