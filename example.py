from explainer import SamplePerturbation, TextExplainer, TabularExplainer 
from adult import UCI_Adult
from imdb import IMDb 
import time
import os
import numpy as np
import argparse
import pickle as pkl


parser = argparse.ArgumentParser()

parser.add_argument("-method", help="type of greedy algorithm", 
                    choices = [ 'faith_shap', 'shapley_taylor', 'shapley_interaction'], 
                    type=str, default='faith_shap')
parser.add_argument("-dataset", choices= ['imdb', 'adult'] , type=str, default="imdb")
parser.add_argument("-T", help="order of interaction", type=int, default = 3)
parser.add_argument("-single_thread", help="Run program with only one cpu", action='store_true')
parser.add_argument("-train_data_path", type=str, default=None , help="load training perturbations in the specified path." )
parser.add_argument("-valid_data_path", type=str, default=None , help="load valid perturbations in the specified path." )
parser.add_argument("-lasso_alpha", type=float, default = 0.001) #0.02 for adults
parser.add_argument("-dump_results_path", type=str, default = None)

args = parser.parse_args()

if args.single_thread:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Load perturbations
data_ids, preds, train_perturbations = pkl.load(open(args.train_data_path,'rb'))

if args.valid_data_path:
    valid_ids, valid_preds, valid_perturbations = pkl.load(open(args.valid_data_path,'rb'))
    assert valid_ids == data_ids
else:
    valid_perturbations = [ None for _ in data_ids]

# Explain
if args.dataset == 'imdb':
    dataset = IMDb()
    explainer = TextExplainer(args.T, args.method, lasso_alpha = args.lasso_alpha)

    results = [] 
    for idx, data_id in enumerate(data_ids):
        print(dataset.raw_testset[data_id])
        length = len(dataset.raw_testset[data_id].split())
        print("Data:", data_id)
        print("Number of training perturbations:", len(train_perturbations[idx][0]))

        print("Length of data:", len(dataset.raw_testset[data_id].split()))
        explanation, train_infd, valid_infd, _ = explainer.explain_instance(dataset.raw_testset[data_id],
                                                                            train_perturbations[idx],
                                                                            valid_perturbations[idx])

        # Show explanation
        explainer.display(explanation, preds[idx][0])
        print("-"*150)
        
        results.append((explanation, data_id))
    if args.dump_results_path is not None:
        pkl.dump(results, open(args.dump_results_path,'wb'))

elif args.dataset == 'adult':
    dataset = UCI_Adult()

    explainer = TabularExplainer(args.T,  args.method, dataset.get_feature_names(), 
                                 lasso_alpha = args.lasso_alpha) 
    L = []
    for idx, data_id in enumerate(data_ids):
        print("Data:", data_id)
        
        explanation, train_infd, valid_infd, _ = explainer.explain_instance(dataset.raw_testset.iloc[data_id],
                                                                            train_perturbations[idx],
                                                                            valid_perturbations[idx])
        # Show explanation
        explainer.display(explanation, preds[idx][0])
        print("-"*50)
        
        results.append((explanation, data_id))
    if args.dump_results_path is not None:
        pkl.dump(results, open(args.dump_results_path,'wb'))
