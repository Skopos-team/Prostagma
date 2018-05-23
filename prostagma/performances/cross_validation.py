
class CrossValidation(object):
	"""docstring for CrossValidation"""
	def __init__(self, k_fold):
		super(CrossValidation, self).__init__()
		"""
			args: 
				k_fold -> int : number of fold to used.
		"""
		self.k_fold = k_fold

	"""
		Search for Iterated K-fold validation with shuffling.
	"""

	def fit(self, X_train, y_train, model):

		all_mae_histories = [] 
		for i in range(k):
		Prepares the validation data: data from partition #k
		 print('processing fold #', i)
		val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
		 partial_train_data = np.concatenate( [train_data[:i * num_val_samples],
		train_data[(i + 1) * num_val_samples:]], axis=0)
		Prepares the training data: data from all other partitions
		  Licensed to <null>
		Predicting house prices: a regression example
		89
		partial_train_targets = np.concatenate( [train_targets[:i * num_val_samples],
		train_targets[(i + 1) * num_val_samples:]], axis=0)
		Builds the Keras model (already compiled)
		 model = build_model()
		history = model.fit(partial_train_data, partial_train_targets,
		validation_data=(val_data, val_targets),
		epochs=num_epochs, batch_size=1, verbose=0) mae_history = history.history['val_mean_absolute_error']
		  all_mae_histories.append(mae_history)
		return all_scores

	def another_method():
		k=4
		num_validation_samples = len(data) // k
		np.random.shuffle(data)
		validation_scores = [] for fold in range(k):
		Selects the validation- data partition
		  validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
		training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
		model = get_model()
		model.train(training_data)
		validation_score = model.evaluate(validation_data) validation_scores.append(validation_score)
		validation_score = np.average(validation_scores)
	Validation score: average of the validation scores of the k folds
  	model = get_model()
	model.train(data)
	test_score = model.evaluate(test_data)