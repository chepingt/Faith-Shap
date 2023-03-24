import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle as pkl
import os
EPS = 1e-6

class UCI_Adult:
    def __init__(self):
        self.dtypes = [
            ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
            ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
            ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
            ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
            ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
        ]
        self.trainset, self.testset, \
          self.y_train, self.y_test = self.load_data()

        self.raw_trainset, self.raw_testset, _, _ = self.load_data(True)
        
        self.dtypes.remove(("Education-Num", "float32"))
        self.dtypes.remove(("fnlwgt", "float32"))


    def load_data(self, raw = False):

        train_set = pd.read_csv('data/adult/adult.data', names=[d[0] for d in self.dtypes], dtype=dict(self.dtypes))
        test_set = pd.read_csv('data/adult/adult.test', skiprows = 1, names=[d[0] for d in self.dtypes], 
                               dtype=dict(self.dtypes)) # Make sure to skip a row for the test set
        
        train_nomissing = train_set.replace(' ?', np.nan).dropna()
        test_nomissing = test_set.replace(' ?', np.nan).dropna()

        test_nomissing['Target'] = test_nomissing['Target'].replace({' <=50K.': ' <=50K', ' >50K.':' >50K'})

        combined_set = pd.concat([train_nomissing, test_nomissing], axis = 0) # Stacks them vertically
        # Remove dtype
        #del combined_set['fnlwgt']
        combined_set = combined_set.drop('Education-Num',axis=1)
        combined_set = combined_set.drop('fnlwgt', axis=1)

        y = combined_set.pop('Target')

        y = pd.Categorical(y).codes

        y_train = y[:train_nomissing.shape[0]]
        y_test = y[train_nomissing.shape[0]:]

        combined_set_dum = pd.get_dummies(combined_set)
        # One-hot encoding
        trainset = combined_set_dum[:train_nomissing.shape[0]] # Up to the last initial training set row
        testset = combined_set_dum[train_nomissing.shape[0]:] # Past the last initial training set row
        
        if raw:
            raw_train = combined_set[:train_nomissing.shape[0]] # Up to the last initial training set row
            raw_test = combined_set[train_nomissing.shape[0]:] # Past the last initial training set row

            return raw_train, raw_test, y_train, y_test
        else: 
            return trainset, testset, y_train, y_test


    def train(self, trainset, y_train):
        xgdmat = xgb.DMatrix(trainset, y_train) # Create our DMatrix to make XGBoost more efficient

        our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                     'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}

        final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)
        return final_gb

    def test(self, testset, y_test, final_gb):
        testdmat = xgb.DMatrix(testset)

        y_pred = final_gb.predict(testdmat) # Predict using our testdmat

        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        acc = accuracy_score(y_pred, y_test)
        return y_pred, acc

    def get_mean_scores(self):
        means = {}
        for k, v in self.dtypes:
            if v == 'float32':
                means[k] = np.mean(self.trainset[k])
        means['Target'] = np.mean(self.y_train)
        return means
        
    def get_feature_names(self, raw = True):
        if raw:
            return [d[0] for d in self.dtypes[:-1]] # The last one is target
        else:
            return [ k for k in self.trainset.columns ]

    def get_model(self):
        if os.path.exists("data/adult/xgboost.pkl"):
            final_gb = pkl.load(open("data/adult/xgboost.pkl", 'rb'))
            _, acc = self.test(self.testset, self.y_test, final_gb)
            print("Accuracy of xgboost on testset:", acc)
        else:
            final_gb = self.train(self.trainset, self.y_train)
            _, acc = self.test(self.testset, self.y_test, final_gb)
            pkl.dump(final_gb, open("data/adult/xgboost.pkl",'wb'))
            print("Accuracy of xgboost on testset:", acc)
        def model(X):
            preds = final_gb.predict(xgb.DMatrix(X))
            #logits = [ np.log(p/(1-p+EPS)) for p in preds]
            return preds
        return model
    
    def get_mapping_function(self):
        D = {k:i for i, (k,v) in enumerate(self.dtypes) }
        means = {}
        variance = {}

        for k, v in self.dtypes:
            if v == 'float32':
                means[k] = np.mean(self.trainset[k])
                variance[k] = np.var(self.trainset[k])

        def data_mapping(binary_vectors, instance):
            # instance : Series in pandas
            raws = []
            for v in binary_vectors:
                x = np.zeros(len(instance))
                for i, k in enumerate(instance.index):
                    if "_" not in k:
                        # float 32
                        index = D[k]
                        if v[index] == 1:
                            x[i] = instance[i]
                        else:
                            #x[i] = means[k]
                            x[i] = np.random.normal(means[k], variance[k])
                    else:
                        #category
                        # 0 -> (0,0,..,0) instead of one-hot encoding
                        category = k.split('_')[0]
                        index = D[category]
                        if v[index] == 1:
                            x[i] = instance[i]
                        else:
                            x[i] = 0
                raws.append(x)
            raws = np.array(raws)
            return pd.DataFrame(raws, columns = self.testset.columns)
        return data_mapping
    
    def get_continuous_data_mapping(self):
        # normalize features to the range of 0 to 1
        maxs = np.zeros(len(self.trainset.columns))
        mins = np.zeros(len(self.trainset.columns))

        for idx, k in enumerate(self.trainset.columns):
            maxs[idx] = self.trainset[k].max()
            mins[idx] = self.trainset[k].min()
        
        n_dim = len(maxs)

        def data_mapping(vecs):
            assert len(vecs[0]) == n_dim
            vecs = vecs * (maxs - mins) + mins
            return pd.DataFrame(vecs, columns = self.testset.columns)
        
        def inv_data_mapping(instances):
            assert len(instances[0]) == n_dim
            n_data = len(instances)
            condition = np.tile((maxs - mins),(n_data, 1))
            vecs = (instances - mins) / (maxs - mins + 1e-10)
            return vecs

        return data_mapping, inv_data_mapping

    def get_gaussian_data_mapping(self):
        #calcualte mean vector
        means = np.zeros(len(self.trainset.columns))
        vars = np.zeros(len(self.trainset.columns))
        
        for idx, k in enumerate(self.trainset.columns):
            means[idx] = self.trainset[k].mean()
            vars[idx] = self.trainset[k].var()

        n_dim = len(means)

        def data_mapping(gaussian_vecs):
            assert len(gaussian_vecs[0]) == n_dim
            vecs = gaussian_vecs * vars + means
            return pd.DataFrame(vecs, columns = self.testset.columns)
        
        def inv_data_mapping(instances):
            assert len(instances[0]) == n_dim
            gaussians = (instances - means) / (vars + 1e-10)
            return gaussians

        return data_mapping, inv_data_mapping


if __name__ == '__main__':
    dataset = UCI_Adult()
    X = dataset.testset['Age']
    Y = dataset.y_test

    D = {}
    for x, y in zip(X,Y):
        if x in D:
            D[x] += [y]
        else:
            D[x] = [y]
    X = []
    Y = []
    for x, y in sorted(D.items(), key=lambda x:x[0]):
        X.append(x)
        Y.append(np.mean(y))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    plt.plot(X,Y)
    plt.savefig("temp.png") 
    #model = dataset.get_model()
