# Prostagma

"Command" - Tell me the parameters and I will give you the best model.

### Main Assumption

A function that builds a Keras model must be passed to the Validator/Technique in order to be able to run the Tuning/Validation.

```python

def build_model(self, parameters=None):        
        
	# Get parameters
	try:
		dropout = parameters["dropout"]
		learning_rate = parameters["learning_rate"]
	except:
		dropout = 0.1
		learning_rate = 0.001

	# Define Model architecture
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding="SAME", input_shape=(28, 28, 1)))
	model.add(Activation("relu"))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(Dropout(dropout))
	model.add(Dense(10))
	model.add(Activation("softmax"))

	# Compile the model
	model.compile(
		optimizer=Adam(learning_rate),
		loss="categorical_crossentropy", 
		metrics=["accuracy"])

	return model

```

### Installation and requirements

The requirements are:

- python>=3.5
- keras>=2.01
- tensorflow>=1.8
- numpy>=1.13.1
- matplotlib>=2.0.2
- scipy>=1.0.0

Then you can clone the repository or install it from pip.

	$ pip install prostagma 

### Usage

```python

import numpy as np

# All the modules to create the Keras Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizers import Adam

# All the modules to define the Search Technique
from prostagma.techniques.grid_search import GridSearch
from prostagma.performances.cross_validation import CrossValidation
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to feed the model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# Rescale Features
X_train = X_train.astype("float32")
X_train /= 255

X_test = X_test.astype("float32")
X_test /= 255

# y to Categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def build_model(self, parameters=None):        
        
	# Get parameters
	try:
		dropout = parameters["dropout"]
		learning_rate = parameters["learning_rate"]
	except:
		dropout = 0.1
		learning_rate = 0.001

	# Define Model architecture
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding="SAME", input_shape=(28, 28, 1)))
	model.add(Activation("relu"))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(Dropout(dropout))
	model.add(Dense(10))
	model.add(Activation("softmax"))

	# Compile the model
	model.compile(
		optimizer=Adam(learning_rate),
		loss="categorical_crossentropy", 
		metrics=["accuracy"])

	return model

def main():

	# Directly Validate the model

	validator = CrossValidation(
		k_fold=3, 
		epochs=10, 
		batch_size=32)
	results = validator.fit(X_train, y_train, build_model)
	print("Mean: %f     Std(%f)" % (results.mean(), results.std()))

	# Tune Parameters

	# Define the dictionary of parameters
	parameters = {
		"dropout" : [0.25, 0.5, 0.75],
		"learning_rate" : [0.1, 0.01, 0.001, 0.0001]
	}

	# Define the Strategy to use
	strategy = GridSearch(
		parameters=parameters, 
		model=build_model, 
		performance_validator=CrossValidation(
			k_fold=3,
			epochs=10,
			batch_size=32
		)
	)
	strategy.fit(X_train, y_train)

	# Show the results
	print("Best Parameters: ")
	print(strategy.best_param)
	print("Best Score Obtained: ")
	print(strategy.best_score)

main()

```
## Performance Validators

Here all the available performance validators with pro and cons.

### Cross Validation

### Hold Out

## Search Technique

Here all the implemented techniques with a brief explanation.

### GridSearch

