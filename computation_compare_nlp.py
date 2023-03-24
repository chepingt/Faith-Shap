import sys, os 
import numpy as np
import math
import itertools
from itertools import chain, combinations, permutations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import argparse
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import random
import time 

from solver import Solver

class Trajectoies():
    def __init__(self, N = 10):
        #self.data = {'l2':[], 'topN_l2':[],'prec':[], 'spr':[], 'values':[]}
        self.data = {'l2':[], 'topN_l2':[],'prec':[], 'spr':[]}
        
        self.checkpoints = []
        self.final_values = {}
        self.N = N
    
    def dict_to_vec(self, D):
        keys = sorted(D.keys())
        v = np.zeros(len(keys))
        for i, k in enumerate(keys):
            v[i] = D[k]
        return v
    
    def update(self, checkpoint, gt_dict, pred_dict, verbose = False):
        l2_distance = 0
        cnt = 0
        nonzero_cnt = 0

        for d in [gt_dict, pred_dict]:
            if 'bias' in d:
                del d['bias']

        for k, v in gt_dict.items():
            l2_distance += (v - pred_dict[k])**2
            cnt += 1
            if abs(v) > 1e-5:
                nonzero_cnt += 1
        l2_distance /= cnt
        
        # Spearman Ranking correlation
        L1 =[]
        L2 = []
        for k, v in gt_dict.items():
            L1.append(abs(v))
            L2.append(abs(pred_dict[k]))
        spearmanr, p_value = stats.spearmanr(L1,L2)
        
        # F1 @ N
        L1 = [(k,v) for k,v in pred_dict.items()]
        L1 = sorted(L1, key = lambda x:abs(x[1]), reverse=True)
        S1 = set( [ k for k,v in L1[:self.N]])
        
        L2 = [(k,v) for k,v in gt_dict.items()]
        L2 = sorted(L2, key = lambda x:abs(x[1]), reverse=True)
        S2 = set( [ k for k,v in L2[:self.N]])
        n_intersec = len(S1.intersection(S2))
        precision = n_intersec / self.N
        if verbose:
            for k, v in L2:
                print(k, pred_dict[k], gt_dict[k])
        
        # top N l2:
        topN_l2_distance = 0
        for k, v in L2[:self.N]:
            topN_l2_distance += (v - pred_dict[k])**2
        topN_l2_distance /= self.N

        self.data['l2'] += [l2_distance]
        self.data['topN_l2'] += [topN_l2_distance]
        self.data['prec'] += [precision]
        self.data['spr'] += [spearmanr]
        #self.data['values'] += [self.dict_to_vec(pred_dict)]
        self.checkpoints += [checkpoint]

        return l2_distance, topN_l2_distance, precision, spearmanr

class CurveData():
    def __init__(self, checkpoints):
        #self.data = {'l2':[], 'topN_l2':[],'prec':[], 'spr':[], 'values':[]}
        self.data = {'l2':[], 'topN_l2':[],'prec':[], 'spr':[]}
        self.checkpoints = checkpoints

    def update(self, traj):
        for k, v in self.data.items():
            v.append(traj.data[k])
    
    def aggregate_data(self):
        self.aggregated_data = {}
        
        for k, v in self.data.items():
            if k == 'values':
                continue
            m, low_p, high_p = self.get_mean_and_std_seqs(v)
            self.aggregated_data[k]=[ {'mean': m[i], 'low_p':low_p[i], 'high_p':high_p[i]} for i in range(len(m))]
        return self.aggregated_data
    
    def get_average_stop_time(self, desired_l2, larger_than_1000 = False):
        stop_evals = []
        l2s = []

        for l2_traj in self.data['l2']:
            stop_evals.append(self.checkpoints[-1]) 
            l2s.append(l2_traj[-1]) 
            for i, x in enumerate(l2_traj):
                if x < desired_l2:
                    if larger_than_1000 and self.checkpoints[i] < 1000:
                        continue 
                    stop_evals[-1] = self.checkpoints[i]
                    l2s[-1] = l2_traj[i]
                    break
                    
        return stop_evals, l2s
    
    def iterator(self):
        ag = self.aggregated_data
        for idx, checkpoint in enumerate(self.checkpoints):
            print(ag)
            print(self.checkpoints)
            L = [checkpoint, ag['l2'][idx], ag['spr'][idx], ag['prec'][idx], ag['topN_l2'][idx]]
            yield L
    
    def get_mean_and_std_seqs(self, seqs):
        # seqs : list of seq of the same length
        # should be correct: return 5% and 95% quantile
        def get_mean_and_std(data):
            a = 1.0 * np.array(data)
            m, std = np.mean(a), np.std(a)
            return m, m-std, m + std
            
        # transpose
        ms = []
        lows = []
        highs = []
        np_seqs = np.array(seqs)

        for values_t in np_seqs.T:
            m, l, h = get_mean_and_std(values_t)
            ms.append(m)
            lows.append(l)
            highs.append(h)
        return np.array(ms), np.array(lows), np.array(highs),  

class Instance:
    def __init__(self, Xs, Ys):
        # checkpoints: a sorted list of intergers denoting the number of samples (used for estimating shapleys)
        self.n_sample, self.d = np.shape(Xs)
        self.D = np.zeros(self.n_sample) # a binary number to its value, e.g. 5 -> '101' -> [1,0,1] -> y
        self.Xs = Xs
        self.Ys = Ys
        self.reset_flag()

        self.multiplier = np.array([ int(np.power(2,i)) for i in range(self.d)])
        for x, y in zip(Xs, Ys):
            temp = np.dot(x, self.multiplier)
            self.D[temp] = y


    def discrete_derivative(self, S, T):
        #input: tuples S and T
        #return: \Delta_S(v(T))
        bin_s = self.to_binary(S)
        bin_t = self.to_binary(T)
        assert int(bin_s + bin_t) == int(bin_s ^ bin_t) # S and T are mutual exclusive
        if len(S) == 0:
            if self.flag[bin_t] == 0:
                self.flag[bin_t] = 1
                self.n_flag += 1
            return self.D[bin_t]
        else:
            keys, signs = self.per(S)
            keys += bin_t
            self.n_flag += len(keys) -  np.sum(self.flag[keys])
            self.flag[keys] = 1
            tmp = self.D[keys] * signs
            return np.sum(tmp)

    def to_binary(self, x):
        if isinstance(x, tuple):
            if len(x) == 0:
                return 0
            return np.sum(self.multiplier[np.array(x)])
        elif isinstance(x, (np.ndarray, np.generic)):
            n_samples = x.shape[0]
            keys = np.matmul(x, self.multiplier.reshape(-1,1)).reshape(-1).astype('int')
            return keys
    
    def reset_flag(self):
        self.n_flag = 0
        self.flag = np.zeros(self.n_sample)  
                
    def per(self,s):
        # s : a tuple of positive numbers
        n = len(s)
        lst = np.array(list(itertools.product([0, 1], repeat=n)))
        signs = np.remainder(np.sum(lst, axis = 1), 2)
        signs = signs * 2 - 1
        if len(s) % 2 == 0:
            signs *= (-1)
        tmp = self.multiplier[np.array(s)].reshape(n,1)
        return np.matmul(lst,tmp).reshape(-1), signs

class ShapleyCalculator:
    def __init__(self, instance, l, checkpoints, lower_orders = True, tracking_loss = True):
        # checkpoints: a sorted list of intergers denoting the number of samples (used for estimating shapleys)
        self.d = instance.d
        self.l = l
        self.I = instance
        self.Xs = instance.Xs
        self.Ys = instance.Ys
        self.checkpoints = checkpoints
        self.tracking_loss = tracking_loss

        if lower_orders:
            self.min_order = 1
        else:
            self.min_order = self.l

        self.I.reset_flag()

    def get_exact_shapleys(self):
        taylor = {}
        interaction = {}
        fac = math.factorial
        total = 0
        for i in range(self.min_order,self.l+1):
            for inter in list(combinations(range(self.d), i)):
                S = tuple(inter)
                s = len(inter)
                interaction[inter] = 0.
                if i < self.l:
                    taylor[inter] = self.I.discrete_derivative(S,()) 
                else:
                    taylor[inter] = 0.

                pow_set = powerset(set(range(self.d)) - set(inter))
                for T in pow_set:
                    t = len(T)
                    T = tuple(T)
                    interaction[inter] += 1/ nCr(self.d-s, t) / (self.d-s+1) * self.I.discrete_derivative(S,T) 
                    if i == self.l:
                        taylor[inter] += 1 / nCr(self.d-1, t) / self.d * i * self.I.discrete_derivative(S,T)
        
        self.I.reset_flag()
        solver = Solver(self.Xs,self.Ys,self.Xs,self.Ys,self.l,is_completeness=True)
        faith, train_infd, valid_infd = solver.solve_faith_shap()
        
        if self.min_order == self.l:
            faith = self.delete_lower_order(faith)

        self.taylor, self.interaction, self.faith = taylor, interaction, faith
        return taylor, interaction, faith
    
    def draw_distribution(self, index):
        #fig, axes = plt.subplots(1, 4,  figsize=(14, 3))            
        
        name = ['taylor','interaction','faith']
        for idx, D in enumerate([self.taylor, self.interaction, self.faith]):
            data = {'absolute value':[]}
            for k, v in D.items():
                data['absolute value'].append(abs(v))
            sns.displot(data, x = 'absolute value')
            plt.tight_layout()
            plt.savefig('figs/value_distribution/{}_{}.png'.format(name[idx], index))
            plt.clf()
    
    def draw_scatter(self, index):
        
        name = ['taylor','interaction','faith']
        functions = [self.get_sampling_Taylor_Shap, self.get_sampling_Interaction_Shap, self.get_sampling_faith_Shap]
        for idx, D in enumerate([self.taylor, self.interaction, self.faith]):
            data = {'True value':[], 'sampling value':[]}
            if idx == 2:
                D_sampling = self.get_sampling_faith_Shap(lasso_alpha = 1e-4).final_values
            else:
                D_sampling = functions[idx]().final_values
            
            for k, v in D.items():
                data['True value'].append(v)
                data['sampling value'].append( D_sampling[k])

            spearmanr, p_value = stats.spearmanr(data['True value'], data['sampling value'])
            data = pd.DataFrame.from_dict(data)
            print(name[idx], index, "Rank correlation:", spearmanr)
            sns.scatterplot(data = data, x = 'True value', y = 'sampling value')
            plt.tight_layout()
            plt.savefig('figs/loss_distribution/{}_{}.png'.format(name[idx],index))
            plt.clf()

    def get_sampling_Interaction_Shap(self, verbose = False):
        cnt_dict = {}
        sum_dict = {}
        traj = Trajectoies()
        eval_cnt = 0
        
        #initialization
        for i in range(self.min_order, self.l+1):
            for inter in list(combinations(range(self.d), i)):
                sum_dict[inter] = 0
                cnt_dict[inter] = 0
        
        now_cpt = 0
        while now_cpt < len(self.checkpoints):
            self.I.reset_flag()
            p = np.random.permutation(range(self.d))

            for j in range(self.min_order, self.l+1):
                for k in range(self.d-j+1):
                    # k, k+1,...,k+j-1
                    inter = tuple(sorted([ p[x+k] for x in range(j) ] ) )
                    sum_dict[inter] += self.I.discrete_derivative(inter, tuple(p[:k])) 
                    cnt_dict[inter] += 1
                    
                    # Update
                    if self.I.n_flag + eval_cnt > self.checkpoints[now_cpt]:
                        temp_d = self.get_value_dict(cnt_dict, sum_dict)
                        if self.tracking_loss:
                            traj.update(self.checkpoints[now_cpt], self.interaction, temp_d)
                        now_cpt += 1
                        if now_cpt >= len(self.checkpoints):
                            break
                if now_cpt >= len(self.checkpoints):
                    break
            eval_cnt += self.I.n_flag

        traj.final_values = self.get_value_dict(cnt_dict, sum_dict)
        
        if verbose:
            self.print_top_interaction(self.interaction, 'True Shapley Interaction')
            self.print_top_interaction(traj.final_values, 'Sampling Shapley Interaction')
        
        return traj
    
    def get_sampling_first_order_shapley(self, lasso_alpha = 0., ridge_alpha = 0.):
        traj = Trajectoies()
        n_eval = self.checkpoints[-1]
        
        # get samples
        X = []
        ps = [ 1/ x / (self.d-x) for x in range(1,self.d)]
        ps = np.array(ps) / np.sum(ps)
        n_elements = np.random.choice(list(range(1,self.d)), n_eval, p = ps)
        n_elements[0] = 0
        n_elements[1] = self.d
        now = 0
        for i in range(n_eval):
            k = n_elements[now]
            now += 1
            s = tuple(np.random.choice(self.d, k, replace = False))
            x = np.zeros(self.d)
            if len(s) > 0:
                x[np.array(s)] = 1
            X.append(x)
        indexes = np.arange(2, n_eval)
        np.random.shuffle(indexes)
        X = np.array(X)
        X[2:] = np.array(X[indexes])
        
        keys = self.I.to_binary(X)
        Y = self.I.D[keys]

        # calculate faith shapley
        now_cpt = 0
        for cpt in self.checkpoints:
            solver = Solver(X[:cpt], Y[:cpt], None, None, 1, is_completeness=True)
            if lasso_alpha > 0:
                D, train_infd, valid_infd = solver.solve_faith_shap(lasso=True, ridge = False, alpha = lasso_alpha)
            elif ridge_alpha > 0:
                D, train_infd, valid_infd = solver.solve_faith_shap(lasso = False, ridge=True, alpha = ridge_alpha)
            else:
                D, train_infd, valid_infd = solver.solve_faith_shap(lasso=False,ridge=False)
            
            if self.tracking_loss:
                traj.update(cpt, self.faith, D)
        traj.final_values = D            
        return traj

    def get_sampling_Taylor_Shap(self, verbose = False):
        cnt_dict = {}
        sum_dict = {}
        traj = Trajectoies()
        eval_cnt = 0
        
        # Calculating exact lower-order values
        for i in range(self.min_order, self.l+1):
            for inter in list(combinations(range(self.d), i)):
                if i <= self.l-1:
                    sum_dict[inter] = self.I.discrete_derivative(inter, tuple())
                    cnt_dict[inter] = 1
                    eval_cnt += math.pow(2, i)
                else:
                    sum_dict[inter] = 0
                    cnt_dict[inter] = 0
        
        now_cpt = 0
        m = 1
        while now_cpt < len(self.checkpoints):
            self.I.reset_flag()
            p = np.random.permutation(range(self.d))
            for inter in list(combinations(range(self.d), self.l)):
                key = tuple(sorted([ p[x]  for x in inter]))
                first_index = key[0]
                sum_dict[key] += self.I.discrete_derivative( key, tuple(p[:inter[0]]))
                cnt_dict[key] += 1
                
                # Update
                if self.I.n_flag + eval_cnt > self.checkpoints[now_cpt]:
                    temp_d = self.get_value_dict(cnt_dict, sum_dict)
                    if self.tracking_loss:
                        traj.update(self.checkpoints[now_cpt], self.taylor, temp_d)
                    now_cpt += 1
                    if now_cpt >= len(self.checkpoints):
                        break
                
            eval_cnt += self.I.n_flag
        
        traj.final_values = self.get_value_dict(cnt_dict, sum_dict)
        
        if verbose:
            self.print_top_interaction(self.taylor, 'True Shapley Taylor')
            self.print_top_interaction(traj.final_values, 'Sampling Shapley Taylor')

        return traj

    def get_sampling_faith_Shap(self, lasso_alpha = 0, ridge_alpha = 0, verbose = False):
        traj = Trajectoies()
        n_eval = self.checkpoints[-1]
        
        # get samples
        X = []
        ps = [ 1/ x / (self.d-x) for x in range(1,self.d)]
        ps = np.array(ps) / np.sum(ps)
        n_elements = np.random.choice(list(range(1,self.d)), n_eval, p = ps)
        n_elements[0] = 0
        n_elements[1] = self.d
        now = 0
        for i in range(n_eval):
            k = n_elements[now]
            now += 1
            s = tuple(np.random.choice(self.d, k, replace = False))
            x = np.zeros(self.d)
            if len(s) > 0:
                x[np.array(s)] = 1
            X.append(x)
        indexes = np.arange(2, n_eval)
        np.random.shuffle(indexes)
        X = np.array(X)
        X[2:] = np.array(X[indexes])
        
        keys = self.I.to_binary(X)
        Y = self.I.D[keys]

        # calculate faith shapley
        now_cpt = 0
        for cpt in self.checkpoints:
            solver = Solver(X[:cpt], Y[:cpt], None, None, self.l, is_completeness=True)
            
            if cpt > 4000 and len(self.checkpoints) > 50:
                #print("cpt > 2000")
                D = D # do nothing       
            elif lasso_alpha > 0:
                D, train_infd, valid_infd = solver.solve_faith_shap(lasso=True, ridge = False, alpha = lasso_alpha)
            elif ridge_alpha > 0:
                D, train_infd, valid_infd = solver.solve_faith_shap(lasso = False, ridge=True, alpha = ridge_alpha)
            else:
                D, train_infd, valid_infd = solver.solve_faith_shap(lasso=False,ridge=False)
             
            if self.min_order == self.l:        
                D = self.delete_lower_order(D)
            
            
            if self.tracking_loss:
                traj.update(cpt, self.faith, D)
        traj.final_values = D            
        
        if verbose:
            self.print_top_interaction(self.faith, 'True Faith Shapley')
            self.print_top_interaction(traj.final_values, 'Sampling Faith Shapley')
        
        return traj
    
    def get_value_dict(self, cnt_d, sum_d):
        value_dict = {}
        for i in range(self.min_order,self.l+1):
            for inter in list(combinations(range(self.d), i)):
                if cnt_d[inter] > 0:
                    value_dict[inter] = sum_d[inter] / cnt_d[inter]
                else:
                    value_dict[inter] = 0
        return value_dict
    
    def print_top_interaction(self, D, name):
        def get_zero_cnt(d):
            cnt = 0
            for k, v in d.items():
                if abs(v) < 1e-7:
                    cnt += 1
            return cnt
        L = []
        for k, v in D.items():
            L.append((k,v))
        L = sorted(L, key = lambda x:abs(x[1]), reverse=True)
        print('-'*50 + name + '-' * 50)
        zero_cnt = get_zero_cnt(D)
        print("# of zero elements: {}/{}".format(zero_cnt, len(D)))
        for k, v in L[:15]:
            print(k,v)

    def delete_lower_order(self, D):
        L = []
        for k, v in D.items():
            if k == 'bias' or len(k) < self.l:
                L.append(k)
        for k in L:
            del D[k]
        return D
                
def powerset(S):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(S)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
                
def load_perturbations(data_path, min_d = 8, max_d = 12):
    data_ids, preds, train_perturbations = pkl.load(open(data_path,'rb'))
    Is = []
    for idx, pert in enumerate(train_perturbations):
        X, Y = pert
        d = np.shape(X)[1]
        if d < min_d or d > max_d:
            continue
        I = Instance(X,Y)
        Is.append(I)
    return Is

def nCr(n,r):	
    if n < r:
        return 0
    value = 1.0
    for i in range(1,r+1):
        value *= (n+1-i) / i
    return value

def plot_curve(l, instances, checkpoints, calculator, plot_std = False, lower_orders = False,
            n_trials = 200, lasso = 0.001, ridge = 0, fig_path = 'temp.png'):
    
    pkl_path = fig_path.replace('.png','.pkl')
    if os.path.exists(pkl_path):
        AllCurveData = pkl.load(open(pkl_path, 'rb'))
    else:
        taylor_curve_data = CurveData(checkpoints) 
        interaction_curve_data = CurveData(checkpoints) 
        faith_curve_data = CurveData(checkpoints) 
        for idx, instance in enumerate(instances):
            start_time = time.time()
            print("-"*50 + 'Instance ' + str(idx) +'/' + str(len(instances))+ '-'*50)
            SC = calculator(instance, l, checkpoints, lower_orders)
            SC.get_exact_shapleys()

            for t in tqdm(range(n_trials)):
                if t == 0:
                    verbose = True
                else:
                    verbose = False
                
                taylor_traj = SC.get_sampling_Taylor_Shap(verbose)
                interaction_traj = SC.get_sampling_Interaction_Shap(verbose)
                faith_traj = SC.get_sampling_faith_Shap(lasso, ridge, verbose)

                taylor_curve_data.update(taylor_traj)
                interaction_curve_data.update(interaction_traj)
                faith_curve_data.update(faith_traj)
            print("Seconds one instance =", time.time() - start_time)	

        taylor_curve_data.aggregate_data()
        interaction_curve_data.aggregate_data()
        faith_curve_data.aggregate_data()

        AllCurveData = [taylor_curve_data, interaction_curve_data, faith_curve_data] 
        pkl.dump(AllCurveData, open(pkl_path, 'wb'))
         
    data = {
        'Interaction index':[],
        'Number of model evaluations':[],
        'Squared distance':[],
        'Squared distance (Top 10)': [],
        'Averaged Precision@10':[],
        "Spearman's rank correlation":[],
    }
    Names = ['Shapley Taylor', 'Shapley Interaction', 'Faith-Shap'] 
    metrics = ['l2','topN_l2','spr', 'prec']
    lower_bounds = {x:{y:[] for y in metrics} for x in Names }
    upper_bounds = {x:{y:[] for y in metrics } for x in Names  }
    n_evals = {x:{y:[] for y in metrics } for x in Names  }

    now = 0
    for shapley_type, curve_data in zip(Names, AllCurveData):
        for i, l2, spr, prec, topN_l2 in curve_data.iterator():
            #if i < 500:
            #    continue
            data['Interaction index'] += [shapley_type]
            data['Number of model evaluations'] += [i]
            data['Squared distance'] += [l2['mean']]
            data['Squared distance (Top 10)'] += [topN_l2['mean']]
            data["Spearman's rank correlation"] += [spr['mean']]
            data['Averaged Precision@10'] += [prec['mean']]
            for m, value in zip(metrics, [l2, topN_l2, spr, prec]):
                lower_bounds[shapley_type][m] += [value['low_p']]
                upper_bounds[shapley_type][m] += [value['high_p']]
                n_evals[shapley_type][m] += [i]

    data = pd.DataFrame.from_dict(data)
    fig, axes = plt.subplots(1, 4,  figsize=(14, 3))            
    
    key = 'Interaction index' 
    #colors = ['blue', 'green', 'red']
    colors = sns.color_palette("colorblind", len(data[key].unique()) )
    palette = { key:colors[i] for i,key in enumerate(data[key].unique()) }
    dash_list = sns._core.unique_dashes(3)
    dashes = { key: dash_list[i] for i, key in enumerate(data[key].unique()) }
    
    sns.lineplot(x = 'Number of model evaluations', y = 'Squared distance', data = data, \
                 hue=key,ax = axes[0],style = key, markers=True, dashes = dashes, palette=palette)
    
    sns.lineplot(x = 'Number of model evaluations', y = 'Squared distance (Top 10)', data = data, \
                 hue=key, ax = axes[1],style = key, markers=True, dashes = dashes, palette=palette)
    
    sns.lineplot(x = 'Number of model evaluations', y = "Spearman's rank correlation", data = data, \
                 hue=key, ax = axes[2], style = key, markers=True, dashes = dashes, palette=palette)
    
    sns.lineplot(x = 'Number of model evaluations', y = "Averaged Precision@10", data = data, \
                 hue=key, ax = axes[3], style = key, markers=True, dashes = dashes, palette=palette)
    
    if plot_std: 
        for idx, m in enumerate(metrics):
            for shapley_type, color in zip(Names, colors):
                axes[idx].fill_between(n_evals[shapley_type][m], lower_bounds[shapley_type][m],  
                                       upper_bounds[shapley_type][m], facecolor=color, alpha=.3)
                    
                #axes[idx].set_xlim(4700, 13100)
    #axes[2].set_xlim(4700, 13100)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()

def temp(fig_path):
    pkl_path = fig_path.replace('.png','.pkl')
    AllCurveData = pkl.load(open(pkl_path, 'rb'))
    desired_l2 = 0.001
    Names = ['Shapley Taylor', 'Shapley Interaction', 'Faith-Shap'] 
    signs = [False, False, False] 
    for shapley_type, curve_data, sign in zip(Names, AllCurveData,signs):
        average, l2s = curve_data.get_average_stop_time(desired_l2, sign)
        print(shapley_type, np.mean(average), np.mean(l2s))
        print(average[:20])
        #print(l2s)
         

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", help="order of interaction", type=int, default = 2)
    parser.add_argument("-data_path", default = 'exp/bank_computation_eff/bank_full_50instance.train', type=str)
    parser.add_argument("-n_trials", type=int, default = 100)
    parser.add_argument("-n_intn", type=int, default = 50)
    parser.add_argument("-lasso", type=float, default = 0 )
    parser.add_argument("-ridge", type=float, default = 0 )
    parser.add_argument("-seed", type=int, default = 2000 )
    parser.add_argument("-min_d", type=int, default = 10)
    parser.add_argument("-max_d", type=int, default = 12)
    parser.add_argument("-exp_dir", type=str, default = 'figs/nlp_new')
    parser.add_argument("-cpt_start", type=int, default = 600)
    parser.add_argument("-cpt_end", type=int, default = 2000)
    parser.add_argument("-cpt_diff", type=int, default = 200)
    parser.add_argument("-lower_orders", help= "Include lower-order interactions.", action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    #test()

    np.random.seed(args.seed)
    fig_name = 'top10_{}instance_d{}_{}_l{}_n_trials{}_lasso{}_ridge{}_seed{}_std3.png'.format(args.n_intn, args.min_d,\
                args.max_d, args.l, args.n_trials, args.lasso, args.ridge, args.seed)
    checkpoints = list(range(args.cpt_start, args.cpt_end+1, args.cpt_diff))
    
    fig_path = os.path.join(args.exp_dir, fig_name)
    print(fig_path) 
    if os.path.exists(fig_path):
        temp(fig_path)
        exit()
    instances = load_perturbations(args.data_path, args.min_d, args.max_d) 
    print("Loaded data successfully.") 

    fig_path = os.path.join(args.exp_dir, fig_name)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    random.shuffle(instances)
    
    if len(instances) > args.n_intn:
        instances = instances[:args.n_intn]
    else:
        print("# data points is less than {}.".format(args.n_intn))

    plot_curve(args.l, instances, checkpoints, ShapleyCalculator, n_trials = args.n_trials, plot_std = True, lower_orders = args.lower_orders, 
                lasso = args.lasso, ridge = args.ridge, fig_path = fig_path)