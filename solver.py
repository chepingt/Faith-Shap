import sys, os 
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, LassoLars, lars_path, Ridge
import time
import math
import itertools
from itertools import combinations, chain
import scipy

def powerset(S):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(S)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

COMPLETENESS_WEIGHT = 1e6

class Solver:
    '''
    Get explanations with differnt methods.
    '''
    def __init__(self, X, Y, valid_X, valid_Y, order_of_interaction, 
                 continuous = False, is_completeness = False):
        self.X = X # M (# samples) * d (d features) array, order 1 and no interaction term
        self.Y = np.array(Y).reshape(-1,1) # array of size M
        self.valid_X = valid_X
        self.valid_Y = valid_Y.reshape(-1,1) if valid_Y is not None else None
        self.T = order_of_interaction # order of interactions
        self.M, self.d = X.shape
        self.continuous = continuous
        self.is_completeness = is_completeness
    
    def solve_faith_shap(self, lasso = False, ridge = False, alpha = 0.001):
        def shap_weight(s,n):
            if s <= 0 or s >= n:
                return 1000000
            return 1
        
        selected_features = []
        for i in range(1, self.T+1):
            for inter in list(itertools.combinations(range(self.d), i)):
                selected_features.append(tuple(inter))
        X = add_features(self.X, None, selected_features, continuous= self.continuous)
        W = [shap_weight(np.sum(x), self.d) for x in self.X]
     
        R_square, weights, bias = solve_linear_regression(X, self.Y, W, lasso = lasso, 
                                                         ridge = ridge, alpha = alpha)
        
        D = self.list2dict(selected_features, weights, bias)
         
        valid_infd = 0
        if self.valid_X is not None:
            valid_r_square, valid_infd = self.show_valid_performance(selected_features, weights, bias)
        train_r_square, train_infd = self.show_train_performance(selected_features, weights, bias)
            
        return D, train_infd, valid_infd
    
    def solve_shapley_taylor(self, lasso = False, lasso_alpha = 0.001):
        D = {}
        for x, y in zip(self.X, self.Y):
            indexes = np.where(x == 1)[0]
            D[tuple(indexes)] = y[0]          
        expl = {} 
        cnt = 1
        for i in range(1, self.T):
            for inter in list(combinations(range(self.d), i)):
                temp = 0 
                for T in powerset(set(inter)):
                    T = tuple(sorted(list(T)))
                    temp += math.pow(-1, len(inter) -len(T)) * D[tuple(T)]
                expl[tuple(inter)] = temp
                cnt += 1
        n_permutation =  0 
        two_to_l = int(math.pow(2, self.T))
        while cnt < len(self.X):
            n_permutation += 1
            for _ in list(combinations(range(self.d), self.T)):
                base = np.where(self.X[cnt] == 1)[0] 
                final = np.where(self.X[cnt + two_to_l-1] == 1)[0]
                inter = set(final) - set(base)
                inter = tuple(inter) 
                temp = 0
                now = 0
                for i in range(0,self.T+1):
                    for j in range(int(scipy.special.comb(self.T, i))):
                        temp_T = np.where(self.X[cnt +now] == 1)[0] 
                        temp += self.Y[cnt + now][0] * math.pow(-1,self.T - i )
                        now += 1
                if inter not in expl:
                    expl[inter] = temp
                else:
                    expl[inter] += temp
                cnt += now
        for k, v in expl.items():
            if len(k) == self.T and isinstance(k, tuple):
                expl[k] = v / n_permutation
            if len(k) == 0:
                expl['bias'] = v
                del expl[k]
        return expl, _, _  
    
    def solve_shapley_interaction(self, lasso = False, lasso_alpha = 0.001):
        expl = {} 
        cnt_D = {}
        cnt = 0
        
        two_to_l = int(math.pow(2, self.T))
        while cnt < len(self.X):
            base = np.where(self.X[cnt] == 1)[0] 
            final = np.where(self.X[cnt + two_to_l-1] == 1)[0]
            inter = set(final) - set(base)
            inter = tuple(inter) 
            temp = 0
            now = 0
            for i in range(0,self.T+1):
                for j in range(int(scipy.special.comb(self.T, i))):
                    temp_T = np.where(self.X[cnt + now] == 1)[0] 
                    temp += self.Y[cnt + now][0] * math.pow(-1, self.T - i )
                    now += 1
            if inter not in expl:
                expl[inter] = temp
                cnt_D[inter] = 1 
            else:
                expl[inter] += temp
                cnt_D[inter] += 1 
            cnt += now
                
        for k, v in expl.items():
            if len(k) == self.T and isinstance(k, tuple):
                expl[k] = v / cnt_D[k]
            if len(k) == 0:
                expl['bias'] = v
                del expl[k]
        return expl, None, None 
    
    def list2dict(self, selected_features, weights, bias):
        D = {}
        for i, feature in enumerate(selected_features):
            D[feature] = weights[i]
        D['bias'] = bias
        return D
    
    def show_train_performance(self, selected_features, weights, bias):
        X = np.ones((len(self.X), 1))
        added_X = add_features(self.X, X, selected_features, continuous= self.continuous)
        train_r_square, train_infd = get_R_square(added_X, self.Y, weights, bias)
        #print("Train R_square :", train_r_square)
        #print("Train Infd :", train_infd)
        return train_r_square, train_infd

    def show_valid_performance(self, selected_features, weights, bias):
        X = np.ones((len(self.valid_X), 1))
        added_X = add_features(self.valid_X, X, selected_features, continuous= self.continuous)
        valid_r_square, valid_infd = get_R_square(added_X, self.valid_Y, weights, bias)
        #print("Valid R_square :", valid_r_square)
        #print("Valid Infd :", valid_infd)
        return valid_r_square, valid_infd 

def get_R_square(X, Y, weights, bias):
    weights = weights.reshape(len(weights), 1)
    bias = bias.reshape((1,1))
    beta = np.concatenate((bias, weights), axis = 0)
    loss = np.matmul(X, beta) - Y
    loss = np.mean(loss * loss)
    variance = np.var(Y)

    R_square = 1 - loss / variance # [batch]

    return R_square, loss

def solve_linear_regression(X, Y, W = None, lasso = False, ridge = False, lasso_weights = None, alpha = 0.001):
    '''
    X : input matrix 
    Y : target vector
    W : sample weight (a vector) 
    '''
    fit_intercept = True
    if lasso:
        if W is not None:
            W = np.array(W) 
            X = np.concatenate((X, np.ones(len(X)).reshape(-1,1)), axis = 1)
            X = np.sqrt(W).reshape(-1,1) * X 
            Y = np.sqrt(W).reshape(-1,1) * Y 
            Y = Y.reshape(-1)
            fit_intercept = False
        if lasso_weights is not None: 
            model = Lasso(alpha= alpha, warm_start = True, fit_intercept = fit_intercept)
            model.coef = lasso_weights
            reg = model.fit(X,Y)
        else:
            reg = LassoLars(alpha=alpha, fit_intercept = fit_intercept, 
                        max_iter = 1000).fit(X, Y)
        if W is None:
            return reg.score(X, Y), reg.coef_, reg.intercept_[0]
        else:
            return reg.score(X, Y), reg.coef_[:-1], reg.coef_[-1]
    elif ridge:
        reg = Ridge(alpha = alpha).fit(X, Y, sample_weight = W)
        return reg.score(X, Y), reg.coef_[0], reg.intercept_[0]

    else:
        reg = LinearRegression().fit(X, Y, sample_weight = W)
        return reg.score(X, Y), reg.coef_[0], reg.intercept_[0]

def add_features(X, ori_X, adding_terms = [], continuous = False):
    '''
    Args:
        X : singleton matrix : (n_perturbations, n_features)
        ori_X : feature matrix before adding an element. (n_perturbation, M)
        adding_term : a list of tuples (e.g. (2,3), (4,5)) or integers (e.g. singletons 3)
    Return:
        tgt_X : feature matrix after adding features in adding_terms. The shape of tgt_X is (n_perturbation, M + len(adding_terms))
    '''
    n_perturbation, d = X.shape
    if ori_X is None:
        tgt_X = np.zeros((n_perturbation, len(adding_terms)))
        M = 0
    else:
        _, M = ori_X.shape
        tgt_X = np.zeros((n_perturbation, M + len(adding_terms)))
        tgt_X[:,:M] = ori_X

    for i, term in enumerate(adding_terms):
        term = np.array(term)
        if term.ndim == 0:
            term = term.reshape(1)

        if continuous:
            tgt_X[:,M+i] = np.prod(X[:,term], axis = 1)
        else:
            tgt_X[:,M+i] = np.all(X[:,term] == 1, axis = 1).astype(int)

    return tgt_X


def nCr(n,r):
    x = 1.0
    for i in range(r):
        x *= (n-i) / float(i+1)
    return int(x)
