from explainer import SamplePerturbation 
from imdb import IMDb
from bank_data import UCI_bank 
import time
import os
import numpy as np
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()

# For generating perturbation:
parser.add_argument("-gpu", help="CUDA_VISIBLE_DEVICES", type=int, default = 1)
parser.add_argument("-dataset", choices= ['imdb',  'bank'] , type=str, default="imdb")
parser.add_argument("-generate_data_path", type=str, default=None , 
                    help="Store perturbation data in the specified path. ($path.train and $path.valid)" )
parser.add_argument("-n_test_data", help="number of test data that you want to explain", type=int, default = 50)
parser.add_argument("-n_train_perturbations", help="number of perturbation (I) for training", type=int, default = 2500)
parser.add_argument("-n_valid_perturbations", help="number of perturbation (I) for validation", type=int, default = 0)
parser.add_argument("-gen_method", help="method of generateing perturbation", 
                    choices= ['banzhaf', 'banzhaf_all', 'faith_shap', 'shapley_taylor', 'shapley_interaction'] , type=str, default="faith_shap")
parser.add_argument("-max_length_limit", help="max length limit for text", type=int, default = 130)
parser.add_argument("-min_length_limit", help="min length limit for text", type=int, default = 0)
parser.add_argument("-n_tabular_average", help="For each perturbation, we \
                    map it to n_tabular_average feature value and then use the average.", type=int, default = 5) 
parser.add_argument("-seed", help="Random seed for numpy. \
                    Note that seeds for generating training and validation set should be different.", type=int, default = 10) 
args = parser.parse_args()

np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.dataset == 'imdb':
    dataset = IMDb()
    istext = True
    args.n_tabular_average = 1
elif args.dataset == 'bank':
    dataset = UCI_bank()
    istext = False
    
model = dataset.get_model()
mapping_function = dataset.get_mapping_function()
feature_names = dataset.get_feature_names()

preds = []
train_perturbations = []
valid_perturbations = []
test_ids = []
now = 0
random_test_ids = np.random.choice(len(dataset.raw_testset), len(dataset.raw_testset), replace = False)

# Start sampling perturbations
generator = SamplePerturbation(model, mapping_function)


while len(test_ids) < args.n_test_data and now < len(random_test_ids):
    test_id = random_test_ids[now]
    now += 1
    if istext:
        if len(dataset.raw_testset[test_id].split()) > args.max_length_limit \
                or len(dataset.raw_testset[test_id].split()) < args.min_length_limit :
            continue
        instance = dataset.raw_testset[test_id]
        n_features = len(instance.split())
        preds.append(model([instance]))
    else:
        instance = dataset.testset.iloc[test_id]
        n_features = len(feature_names)
        preds.append(model(mapping_function(np.ones(n_features).reshape(1,-1), instance)))
    
    X_train, Y_train = generator.sample_perturbations(args.n_train_perturbations, n_features, 
                                                 instance, args.gen_method, 
                                                 n_tabular_average = args.n_tabular_average)

    train_perturbations.append((X_train, Y_train))
    
    if args.n_valid_perturbations > 0:
        X_valid, Y_valid = generator.sample_perturbations(args.n_valid_perturbations, n_features, 
                                                        instance, args.gen_method, 
                                                        n_tabular_average = args.n_tabular_average)
        valid_perturbations.append((X_valid, Y_valid))
    
    test_ids.append(test_id)

pkl.dump((test_ids, preds, train_perturbations), open(args.generate_data_path + '.train', 'wb'))

if args.n_valid_perturbations > 0:
    pkl.dump((test_ids, preds, valid_perturbations), open(args.generate_data_path + '.valid', 'wb'))
