import sys, os
import numpy as np
import time
import math
import itertools
from solver import Solver
from ast import literal_eval as make_tuple
from itertools import combinations, chain
import scipy

def powerset(S):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(S)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class SamplePerturbation():
    """Generate perturbation via monte-carlo sampling """
    def __init__(self, model, data_mapping_function):
        """
        model : maps feature vector to a real value
        data_mapping_function: the mapping function, which takes a binary perturbation as input annd output the original feature vector
        """
        self.model = model
        self.data_mapping_function = data_mapping_function # a function that maps binary vectors to correspondin input feature of the model

    def sample_perturbations(self, n_perturbation, n_features, instance,
                             method = 'banzhaf', n_tabular_average = 1,
                             interaction_order = 2):
        '''
        Args:
        n_perturbation : # of perturbation
        n_features (d) : dimension of feature (dimension of I)
        instance : the input feature of the instance you want to explain
        n_tabular_average : For each perturbation, we map it to n_tabular_average
                            feature values and then use the average."
        methods:
            banzhaf_all: 2^d possible
            banzhaf: uniform sampling
            faith_shap: sample weights according to
            shapley_taylor: permutation-based sampling 
            shapley_interaction: permutation-based sampling 
        '''
        s = time.time()
        Is = self.generate_Is(n_perturbation, n_features, method = method,
                              interaction_order = interaction_order)

        Is_aug = np.repeat(Is, n_tabular_average, axis=0)
        Xs_aug = self.data_mapping_function(Is_aug, instance)
        Ys_aug = self.model(Xs_aug)
        if n_tabular_average > 1:
            Ys = []
            assert len(Ys_aug) == len(Is) * n_tabular_average
            for i in range(len(Is)):
                Ys.append(np.mean([Ys_aug[i*n_tabular_average + j] for j in range(n_tabular_average)]) )
            assert len(Ys) == len(Is)
        else:
            Ys = Ys_aug

        print("Time of generateing perturbation (s):",time.time() - s)
        return Is, np.array(Ys)

    def generate_Is(self, n_perturbation, n_features, method = 'banzhaf',
                    interaction_order = 2):
        Is = []
        #print(method)
        if method == 'banzhaf_all' or n_perturbation == math.pow(2,n_features):
            for v in itertools.product([0, 1], repeat=n_features):
                Is.append(v)
        elif method == 'banzhaf':
            for _ in range(n_perturbation):
                v = np.random.randint(2, size=n_features)
                Is.append(v)
        elif method == 'faith_shap':
            d = n_features
            ps = [ 1/ x / (d-x) for x in range(1,d)]
            ps = np.array(ps) / np.sum(ps)
            n_elements = np.random.choice(list(range(1,d)), n_perturbation, p = ps)
            n_elements[0] = 0
            n_elements[1] = d
            now = 0
            for i in range(n_perturbation):
                k = n_elements[now]
                now += 1
                s = tuple(np.random.choice(d, k, replace = False))
                x = np.zeros(d)
                if len(s) > 0:
                    x[np.array(s)] = 1
                Is.append(x)
        elif method == 'shapley_taylor':
            d = n_features
            l = interaction_order
            Is.append(np.zeros(d))
            cnt = 0
            
            for i in range(1,l):
                for inter in list(combinations(range(d), i)):
                    x = np.zeros(d)
                    x[np.array(inter)] = 1  
                    Is.append(x)
                    cnt += 1
            total = cnt + int(scipy.special.comb(d, l))

            n_permutation = int((n_perturbation - cnt) / total) + 1
            #print("Number of permutation:", n_permutation)
            for _ in range(n_permutation):
                p = np.random.permutation(range(d))
                for inter in list(combinations(range(d), l)):
                    key = tuple(sorted([ p[x] for x in inter]))
                    first_index = key[0]
                    base = p[:inter[0]]
                    for T in powerset(set(key)):
                        x = np.zeros(d)
                        x[base] = 1
                        if len(T) > 0:
                            x[np.array(tuple(T))] = 1
                        Is.append(x)
            Is = np.array(Is)
            return np.array(Is)
        elif method == 'shapley_interaction':
            d = n_features
            l = interaction_order
            total = d-1 + d-2 # only for l=2, number of consecutive number plus one-gapper 

            n_permutation = int(n_perturbation / total) + 1
            #print("Number of permutation:", n_permutation)
            for _ in range(n_permutation):
                p = np.random.permutation(range(d))
                for k in range(d - l + 1):
                    inter = tuple(sorted([ p[x+k] for x in range(l) ] ) )
                    base = p[:k]
                    for T in powerset(set(inter)):
                        x = np.zeros(d)
                        x[base] = 1
                        if len(T) > 0:
                            x[np.array(tuple(T))] = 1
                        Is.append(x)
            Is = np.array(Is)
            return np.array(Is)
        
        Is = np.array(Is)
        np.random.shuffle(Is)
        return Is

class Explainer(object):
    def solve(self, train_perturbation, valid_perturbation = None, 
                    continuous = False, is_completeness = False):

        Is = np.array(Is)
        np.random.shuffle(Is)
        return Is

class Explainer(object):
    def solve(self, train_perturbation, valid_perturbation = None, 
                    continuous = False, is_completeness = False):
        Is, Ys = train_perturbation
        valid_Is, valid_Ys = None, None

        if valid_perturbation:
            valid_Is, valid_Ys = valid_perturbation

        solver = Solver(Is, Ys, valid_Is, valid_Ys, self.order_of_interaction, 
                        continuous = continuous, is_completeness = is_completeness)

        if self.method == 'faith_shap':
            expl, train_infd, valid_infd = solver.solve_faith_shap(self.lasso, alpha = self.lasso_alpha)
        elif self.method == 'shapley_taylor':
            expl, train_infd, valid_infd = solver.solve_shapley_taylor()
        elif self.method == 'shapley_interaction':
            expl, train_infd, valid_infd = solver.solve_shapley_interaction()

        return expl, train_infd, valid_infd

    def display(self, explanation, pred, instance = None, show_product = False):
        if pred != -1:
            print("Output probability:", pred)
        if show_product:
            print("{:87s}  {:8s} | {:20s}".format("Feature (interactions)", "Product", "Feature importance"))
        else:
            print("{:90s} | {}".format("Feature (interactions)", "Feature importance"))
        
        total_product = 0
        total = 0
        for idx, (k,v) in enumerate(explanation):
            #if abs(v) < 0.00004 and idx >= 10:
            #    break
            #if idx >=25:
            #    break
            if show_product:
                product = 1
                if isinstance(k, tuple):
                    if len(k) > 1:
                        temp = []
                        for x in k:
                            name, _ = x.split(':')
                            name = name.rstrip()
                            temp.append("{} : {:.3f}".format(name, instance[name]))
                            product *= instance[name]
                        k = tuple(temp)
                    else:
                        if ':' in k[0]:
                            k = k[0].split(':')[0]
                        else:
                            k = k[0]
                        product = instance[k]
                else:
                    if k != 'bias':
                        product = instance[k]
                print("{:90s}  {:.3f} : {:.5f}".format(str(k), product, v))
                if idx <=7:
                  total_product += float(product) * float(v)
            else:
                print("{:90s} : {:.5f}".format(str(k), v))
            total += v
        if show_product:
            print("Product:", total_product)
        print("Sum:", total)
        return total_product

class TextExplainer(Explainer):
    def __init__(self, order_of_interaction, method,
                 lasso = True, lasso_alpha= 0.001):
        self.order_of_interaction = order_of_interaction
        self.method = method
        self.lasso = lasso
        self.lasso_alpha= lasso_alpha

    def explain_instance(self, instance, train_perturbation,
                         valid_perturbation = None):
        '''
        Args:
            instance : Input text, which is a string.
        Return:
            explaination : a dict of features and their corresponding importance value
            train_infd : infidelity evaluated on the training perturbation
            valid_infd : infidelity evaluated on the validation perturbation
            raw_expl : features are stored as indexes instaed of feature names
        '''
        raw_expl, train_infd, valid_infd = self.solve(train_perturbation, valid_perturbation)

        names = []

        instance = instance.split()

        for k, v in raw_expl.items():
            if type(k) is int:
                names.append((instance[k] + " " + str(k), v))
            elif type(k) is tuple:
                temp = tuple([ instance[i]  + " " + str(i) for i in k])
                names.append((temp, v))
            elif k == 'bias':
                names.append(('bias', v))

        explaination = sorted(names, key = lambda x:abs(x[1]), reverse = True)

        return explaination, train_infd, valid_infd, raw_expl

class TabularExplainer(Explainer):
    def __init__(self, order_of_interaction, method, feature_names,
                 lasso = True, lasso_alpha= 0.02 ):
        self.order_of_interaction = order_of_interaction
        self.method = method
        self.feature_names = feature_names
        self.lasso = lasso
        self.lasso_alpha= lasso_alpha

    def explain_instance(self, raw_instance, train_perturbation,
                         valid_perturbation = None):
        '''
        Args:
            raw_instance : a instance in the format of panda.dataframe ,
                           which maps feature names to raw feature value,
                           i.e. country -> U.S.
        Return:
            explaination : a dict of features and their corresponding importance value
            train_infd : infidelity evaluated in the training perturbation
            valid_infd : infidelity evaluated in the validation perturbation
            raw_expl : features are stored as indexes instead of feature names
        '''
        raw_expl, train_infd, valid_infd = self.solve(train_perturbation, valid_perturbation)
        names = []

        for k, v in raw_expl.items():
            if type(k) is int:
                names.append((self.feature_names[k] + ': {}'.format(raw_instance[self.feature_names[k]]), v))
            elif type(k) is tuple:
                temp = tuple([self.feature_names[i] + ': {}'.format(raw_instance[self.feature_names[i]]) for i in k])
                names.append((temp, v))
            elif k == 'bias':
                names.append(('bias', v))

        explaination = sorted(names, key = lambda x:abs(x[1]), reverse = True)
        return explaination, train_infd, valid_infd, raw_expl
