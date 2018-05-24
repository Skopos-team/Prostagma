import argparse

from project.data_preprocessing.preprocessing import Preprocessor
from project.data_preprocessing.data_loader import Loader
from project.models.model import Model

from prostagma.techniques.evolutionary import EvolutionaryStrategy
from prostagma.performances.cross_validation import CrossValidation

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--restore', type=bool, default=True, 
    help='if True restore the model from --model_path.')
parser.add_argument('--test', type=bool, default=False, 
    help='if True test the model.')
parser.add_argument('--train', type=bool, default=True, 
    help='if True train the model.')

""" Model Parametes """
parser.add_argument('--log_dir', type=str, default='./tensorbaord', 
    help='directory where to store tensorbaord values.')
parser.add_argument('--model_path', type=str, 
    default='./project/weights/model', 
    help='model checkpoints directory.')
parser.add_argument('--epochs', type=int, default=50, 
    help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=32, 
    help='number of samples in the training batch.')
parser.add_argument('--number_of_samples', type=int, default=1500, 
    help='number of samples to load in memory.')
parser.add_argument('--train_samples', type=int, default=1000, 
    help='number of samples to load in memory.')
parser.add_argument('--val_samples', type=int, default=50, 
    help='number of samples to load in memory.')
parser.add_argument('--test_samples', type=int, default=100, 
    help='number of samples to load in memory.')

""" Model Hyperparameters """
parser.add_argument('--learning_rate', type=float, default=0.001, 
    help='learnining rate for the optimizer.')
parser.add_argument('--dropout', type=float, default=0.1, 
    help='percentage of neurons to discard during training.')

args = parser.parse_args()

def main():

    # Remove Tensorboard Folder
    try:
        shutil.rmtree('./tensorbaord')
    except FileNotFoundError:
        pass
    
    # Fix the seed
    np.random.seed(0)

    # Load the data
    loader = Loader(number_of_samples=args.number_of_samples)
    X, y = loader.load_data()
    print("Loaded the data...")

    # Preprocess the data
    preprocessor = Preprocessor(
    	train_samples=args.train_samples, 
    	test_samples=args.test_samples, 
    	val_samples=args.val_samples)
    X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.fit_transform(X, y)

    # Define Default Parameters
    default_parameters = {
		"dropout" : args.dropout,
		"learning_rate" : args.learning_rate
	}

    # Define the Model
    model = Model(
        log_dir=args.log_dir,
        model_path=args.model_path,
        parameters=default_parameters
        )

    # Restore the model
    if args.restore == True:
        model.restore()

    # Train the model
    if args.train == True:
        history = model.fit(X_train, y_train, X_val, y_val, 
        	batch_size=args.batch_size, epochs=args.epochs)

    # Test the model
    if args.test == True:
        model.evaluate(X_test, y_test)

    ####### TUNING #########

    # Define the parameters
    parameters = {
    	"dropout" : [0.25, 0.5, 0.75],
    	"learning_rate" : [0.1, 0.01, 0.001, 0.0001]
    }

    # Find best parameters
    if args.tune == True:
	    strategy = EvolutionaryStrategy(
	    	parameters=parameters, 
	    	model=Model(), 
	    	performance_validator=CrossValidation(
	    			epochs=args.epochs,
	    			batch_size=args.batch_size
	    		)
	    )
	    history, best_param = strategy.fit(X_train, y_train)
    
    return

main()