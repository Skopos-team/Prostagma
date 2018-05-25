

class SearchTechnique(object):
	"""docstring for Technique"""
	def __init__(self, parameters, model, 
			performance_validator=CrossValidation(k=5)):
		super(SearchTechnique, self).__init__()
		"""
			args:
				parameters -> dict : with the parameters to search.
				performance_validator -> Performance object : how to validate 
					the scores [default CrossValidation() with 5 fold]
				model -> function() : method to create the Keras model.
		"""
		self.parameters = parameters
		self.model = model
		self.performance_validator = performance_validator

	def fit(self, X_train, y_train):
		raise NotImplementedError
		