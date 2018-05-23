
class CrossValidation(object):
	"""docstring for CrossValidation"""
	def __init__(self, k_fold):
		super(CrossValidation, self).__init__()
		"""
			args: 
				k_fold -> int : number of fold to used.
		"""
		self.k_fold = k_fold

	def fit(self, X_train, y_train, model):

		num_val_samples = len(train_data) // k 
		num_epochs = 100
		all_scores = []
		for i in range(k):
			print('processing fold #', i)
			val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
		 	
		 	partial_train_data = np.concatenate( [train_data[:i * num_val_samples],
				train_data[(i + 1) * num_val_samples:]], axis=0)
			
			partial_train_targets = np.concatenate( [train_targets[:i * num_val_samples],
				train_targets[(i + 1) * num_val_samples:]], axis=0)
			
			model = build_model()
			
			model.fit(partial_train_data, partial_train_targets, 
				epochs=num_epochs, batch_size=1, verbose=0)
			
			val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
		 	
		 	all_scores.append(val_mae)
		return all_scores
		