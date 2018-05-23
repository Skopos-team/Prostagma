


class GridSearch(object):
	"""
		The class implement the simple Grid Search 
		algorithm to find the best parameters using 
		Cross Validation. 
	"""
	def __init__(self, parameters, model, 
			performance_validator=CrossValidation(k=5)):
		super(GridSearch, self).__init__()
		"""
			args:
				parameters -> dict : with the parameters to search.
				performance_validator -> Performance object : how to validate 
					the scores [default CrossValidation() with 5 fold]
				model -> function() : method to create the Keras model.
		"""
		self.parameters = parameters
		self.performance_validator = performance_validator
		self.model = model


		