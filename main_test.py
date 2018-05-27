import numpy as np

from prostagma.techniques.grid_search import GridSearch
from prostagma.performances.cross_validation import CrossValidation
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

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

	model.add(Conv2D(64, (3, 3), padding="SAME"))
	model.add(Activation("relu"))
	model.add(Conv2D(64, (3, 3)))
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
	    k_fold=5, 
	    epochs=100, 
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
                k_fold=5,
    			epochs=args.epochs,
    			batch_size=args.batch_size
    		)
    )
    strategy.fit(X_train, y_train)

    # Show the results
    print("Best Parameters: ")
    print(strategy.best_param)
    print("Best Score Obtained: ")
    print(strategy.best_score)

main()