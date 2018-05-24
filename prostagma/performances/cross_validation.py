
from prostagma.performances.performance import Performance

class CrossValidation(Performance):
	"""docstring for CrossValidation"""
	def __init__(self, k_fold=5, epochs, batch_size):
		super(CrossValidation, self).__init__(
			epochs=epochs, batch_size=batch_size)
		"""
			args: 
				k_fold -> int : number of fold to used.
		"""
		self.k_fold = k_fold

	"""
		Search for: 
			Iterated K-fold validation with shuffling.
			Stratified K-fold Cross validation.
	"""

	def fit(self, X_train, y_train, network):
		all_mae_histories = [] 
		num_val_samples = len(data) / self.k
		for i in range(self.k):
			print('processing fold #', i)
			val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
			val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
			partial_train_data = np.concatenate( [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
			partial_train_targets = np.concatenate( [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
			history = network.model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0) 
			mae_history = history.history['val_mean_absolute_error']
			all_mae_histories.append(mae_history)
		return all_scores

	def another_method():
		num_validation_samples = len(data) / self.k
		np.random.shuffle(data)
		validation_scores = [] 
		for fold in range(self.k):
			validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
			training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
			network.model.fit(training_data)
			validation_score = network.model.evaluate(validation_data) 
			validation_scores.append(validation_score)
		validation_score = np.average(validation_scores)

