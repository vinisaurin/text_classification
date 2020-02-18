#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:11:36 2020

@author: viniciussaurin
"""
from gensim import utils
import gensim.parsing.preprocessing as gsp
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm
import multiprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics as mtc
import pandas as pd
from sklearn.metrics import confusion_matrix


def clean_text(text, filters):
    text = text.lower()
    text = utils.to_unicode(text)
    for f in filters:
        text = f(text)
    return text

def plot_word_cloud(df, filters, category=None, **kwargs):
    
    if category is not None:
        df = df[df.category==category]
    texts = ''
    for index, item in df.iterrows():
        texts = texts + ' ' + clean_text(item['text'],filters)
        
    wordcloud_instance = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords=None,
                min_font_size = 10).generate(texts) 
             
    plt.figure(figsize = (12,12), facecolor = None) 
    plt.imshow(wordcloud_instance) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    if category is not None:
        plt.title(category, fontsize=20)
    plt.savefig(kwargs['file_name'], transparent=True, bbox_inches='tight')
    plt.show()

class Doc2VecTransformer(BaseEstimator):

    def __init__(self, filters, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1
        self.filters = filters

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(clean_text(row, self.filters).split(), [index]) for index, row in enumerate(df_x)]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(clean_text(row,self.filters).split())
                                     for index, row in enumerate(df_x)]))
    
class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self, filters):
        self._model = TfidfVectorizer()
        self.filters = filters
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x : clean_text(x, self.filters))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)



def roc_graph(model, X_test, y_test, **kwargs):

    labels, y_test_converted = np.unique(y_test, return_inverse=True)
    y_predict_proba = model.predict_proba(X_test)
    n_classes = 5
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = list(map(lambda x: 1 if x == i else 0, y_test_converted))
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])
    
    
    # Plot average ROC Curve
    plt.figure(figsize=(20,12))
    plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["average"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot each individual ROC curve
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class: ' + kwargs['model_name'])
    plt.legend(loc="lower right")
    plt.savefig(kwargs['file_name'], bbox_inches='tight')
    plt.show()
    
    


def conf_matrix(modelo, X, y, labels, save=True, normalize='true', title='Confusion matrix: ', **kwargs):
    
    #np.set_printoptions(precision=2)
    
    disp = plot_confusion_matrix(modelo, X, y,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 values_format='.2f',
                                 xticks_rotation=35.0)
    disp.ax_.set_title(title, fontsize=20)
    disp.ax_.set_xlabel('Predicted label', fontsize=14)
    disp.ax_.set_ylabel('True label', fontsize=14)
    disp.figure_.set_size_inches(12,6)

    print(str(title+kwargs['file_name']))
    print(disp.confusion_matrix)
    
    if save:     
        plt.savefig(str('confusion_matrix_' + kwargs['file_name']), bbox_inches='tight')
        
    return disp.confusion_matrix
        
        

def summary_stats(modelos, X, y, labels, average='weighted'):
    
    df = pd.DataFrame()
    for modelo in modelos.items():
    
        cm = confusion_matrix(y, modelo[1].predict(X))
        
        fneg = cm.sum(axis=1)-cm.diagonal()
        fpos = cm.sum(axis=0)-cm.diagonal()
        tpos = cm.diagonal()
        tneg = cm.sum() - (tpos + fpos + fneg)
        
        # False positive rate
        fpr = fpos/(fpos+tneg)
        # False negative rate 
        fnr = fneg/(fneg+tpos)
        
    
        df1 = pd.DataFrame({
            "Accuracy": [mtc.accuracy_score(y, modelo[1].predict(X))],
            "Precision": [mtc.precision_score(y, modelo[1].predict(X), average=average)],
            "Recall": [mtc.recall_score(y, modelo[1].predict(X), average=average)],
            "F1-score": [mtc.f1_score(y, modelo[1].predict(X), average=average)],
            "Jaccard score": [mtc.jaccard_score(y,modelo[1].predict(X), average=average)],
            "False Positive Rate": fpr.mean(),
            "False Negative Rate": fnr.mean()
        }, index = [modelo[0]])
        df = df.append(df1)
        
    return df









