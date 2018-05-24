
from prostagma.techniques.technique import SearchTechnique

class GridSearch(SearchTechnique):
	"""
		The class implement the simple Grid Search 
		algorithm to find the best parameters using 
		Cross Validation. 
	"""
	def __init__(self, parameters, model, 
			performance_validator=CrossValidation(k=5)):
		super(GridSearch, self).__init__(parameters=parameters, 
			model=model, performance_validator=performance_validator)



		