

class HoldOut(object):
	"""docstring for HoldOut"""
	def __init__(self, arg):
		super(HoldOut, self).__init__()
		self.arg = arg

	def fit(X_train, y_train):
		num_validation_samples = 10000
np.random.shuffle(data)
Defines the validation set
   validation_data = data[:num_validation_samples]
 data = data[num_validation_samples:]
training_data = data[:]
Defines the training set
   model = get_model() Trains a model on the training
model.train(training_data)
validation_score = model.evaluate(validation_data)
# At this point you can tune your model,
# retrain it, evaluate it, tune it again...
data, and evaluates it on the validation data
 model = get_model() Once you’ve tuned your
model.train(np.concatenate([training_data,
                            validation_data]))
test_score = model.evaluate(test_data)
		return
		