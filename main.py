import argparse
import shutil
import numpy as np

from project.data_preprocessing.preprocessing import Preprocessor
from project.data_preprocessing.data_loader import Loader
from project.models.model import Model

from prostagma.techniques.evolutionary import EvolutionaryStrategy
from prostagma.techniques.grid_search import GridSearch
from prostagma.performances.cross_validation import CrossValidation

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--test', type=bool, default=False, 
    help='if True test the model.')
parser.add_argument('--train', type=bool, default=False, 
    help='if True train the model.')

""" Model Parameters """
parser.add_argument('--log_dir', type=str, default='./tensorbaord', 
    help='directory where to store tensorbaord values.')
parser.add_argument('--model_path', type=str, 
    default='./project/weights/model', 
    help='model checkpoints directory.')
parser.add_argument('--epochs', type=int, default=10, 
    help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=32, 
    help='number of samples in the training batch.')
parser.add_argument('--number_of_samples', type=int, default=1500, 
    help='number of samples to load in memory.')
parser.add_argument('--train_samples', type=int, default=50, 
    help='number of samples to load in memory.')
parser.add_argument('--val_samples', type=int, default=50, 
    help='number of samples to load in memory.')
parser.add_argument('--test_samples', type=int, default=100, 
    help='number of samples to load in memory.')

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

    # Define the Model
    model = Model(
        log_dir=args.log_dir,
        model_path=args.model_path
        )

    # Directly Validate the model
    validator = CrossValidation(
        k_fold=5, 
        epochs=args.epochs, 
        batch_size=args.batch_size)
    results = validator.fit(X_train, y_train, model.build_model)
    print("Mean: %f     Std(%f)" % (results.mean(), results.std()))

    # Tuning The Parameters

    # Define the dictionary of parameters
    parameters = {
    	"dropout" : [0.25, 0.5, 0.75],
    	"learning_rate" : [0.1, 0.01, 0.001, 0.0001]
    }

    # Define the Strategy to use
    strategy = GridSearch(
    	parameters=parameters, 
    	model=model.build_model, 
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
    
    return

main()