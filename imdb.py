#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ktrain
from ktrain import text
import pickle as pkl
import numpy as np
import os
EPS = 1e-7
from nltk import tokenize

class IMDb:
    def __init__(self, two_sentence = True):
        if two_sentence:
            self.dir_path = 'data/Imdb_2sentence'
        else:
            self.dir_path = 'data/aclImdb'
        self.model_path = os.path.join(self.dir_path, 'models/aclImdb_bert')
        self.proc_data_path = os.path.join(self.dir_path, "proc_data.pkl")
        self.proc_data_raw_path = os.path.join(self.dir_path, "proc_data_raw.pkl")
        if not os.path.exists(os.path.join(self.dir_path,'models')):
            os.makedirs(os.path.join(self.dir_path,'models'))

        self.raw_trainset, self.raw_testset, \
            self.label_train, self.label_test = self.load_data(True)
        
        self.trainset, self.testset, \
          self.y_train, self.y_test, self.preproc = self.load_data()

    def load_data(self, raw = False):
        if raw:
            if os.path.exists(self.proc_data_raw_path):
                texts_train, label_train, texts_test, label_test = pkl.load(open(self.proc_data_raw_path,'rb')) 
            else:
                texts_train, label_train = self.load_imdb('train')
                texts_test, label_test = self.load_imdb('test')
                pkl.dump( (texts_train, label_train, texts_test, label_test), open(self.proc_data_raw_path,'wb'))
            return texts_train, texts_test, label_train, label_test
        else:
            if os.path.exists(self.proc_data_path):
                (x_train, y_train), (x_test, y_test), preproc = pkl.load(open(self.proc_data_path,'rb'))
            else:
                (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(self.dir_path, maxlen=500, 
                                                                                       preprocess_mode='bert', 
                                                                                       train_test_names=['train', 'test'],
                                                                                       classes=['pos', 'neg'])
                pkl.dump( ((x_train, y_train), (x_test, y_test), preproc), open(self.proc_data_path,'wb'))

            return x_train, x_test, y_train, y_test, preproc
        
    def load_imdb(self, mode = 'train'):
        texts = []
        labels = []
        path = os.path.join(self.dir_path, mode) 
        for label, flag in enumerate(['pos','neg']):
            flag = os.path.join(path, flag)
            for filename in os.listdir(flag):
                with open(os.path.join(flag, filename), 'r', encoding="utf-8") as f:
                    text = f.read()
                    text = text.replace("<br />", " ").rstrip()
                    print(text + '\n')
                    texts.append(text)
                    labels.append(label)
        print("Sucessfully loading.")
        return texts, labels

    def train(self):
        model = text.text_classifier('bert', (self.trainset, self.y_train) , preproc=self.preproc)
        learner = ktrain.get_learner(model, train_data=(self.trainset, self.y_train), val_data=(self.testset, self.y_test), batch_size=4)
        
        # find good learning rate
        learner.lr_find()             # briefly simulate training to find good
        # learning rate
        learner.lr_plot()             # visually identify best learning rate

        # train using 1cycle learning rate schedule for 2 epochs
        learner.fit_onecycle(2e-5, 2) 

        predictor = ktrain.get_predictor(learner.model, self.preproc)

        predictor.save(self.model_path)
        return predictor

    def test(self):
        predictor = ktrain.load_predictor(self.model_path)

        pred_train = predictor.predict_proba(self.raw_trainset)
        pred_test = predictor.predict_proba(self.raw_testset)

        pred_train = process_pred(pred_train)
        pred_test = process_pred(pred_test)

        pkl.dump(pred_train, open(os.path.join(self.dir_path, 'pred_train.pkl'),"wb"))
        pkl.dump(pred_test, open(os.path.join(self.dir_path, "pred_test.pkl"),"wb"))

        print("ACC in training set:", compute_acc(pred_train, self.label_train))
        print("ACC in testing set:", compute_acc(pred_test, self.label_test))

    def get_mean_scores(self):
        return None
        
    def get_feature_names(self):
        return None # The last one is target

    def get_model(self):
        
        if os.path.exists(self.model_path):
            predictor = ktrain.load_predictor(self.model_path)
        else:
            predictor = self.train()

        def model(X):
            # X list of texts
            #return np.array([ np.log(x[1]/(1-x[1]+EPS)) for x in predictor.predict_proba(X)])
            return np.array([ x[1] for x in predictor.predict_proba(X)])

        return model
    
    def get_mapping_function(self):
        def data_mapping(binary_vectors, instance):
            # instance : texts
            words = instance.split(' ')
            texts = []
            for v in binary_vectors:
                text = []
                for idx, x in enumerate(v):
                    if x == 1:
                        text.append(words[idx])
                texts.append(' '.join(text))
            return texts
        return data_mapping

def compute_acc(pred, y):
    total = 0
    correct = 0
    for p, yy in zip(pred,y):
        if p >= 0.5:
            p = 1 
        else:
            p = 0
        if p == yy:
            correct += 1
        total += 1
    return correct / total

def process_pred(pred):
    L = []
    for p in pred:
        L.append(p[0])
    return L


if __name__ == '__main__':
    dataset = IMDb()
    #dataset.train()
    dataset.test()

