import logging
from typing import Dict, List
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def split_data(data:pd.DataFrame) -> List:
    x=data['tweet']
    y=data['label']
    train_x, valid_x, train_y, valid_y = train_test_split(x,y)
    return [train_x, valid_x, train_y, valid_y]

def vector(data: pd.DataFrame, parameters: Dict, train_x: pd.DataFrame, valid_x: pd.DataFrame) -> TfidfVectorizer:
    tfidf_vect = TfidfVectorizer(analyzer=parameters['analyzer'], token_pattern = parameters['token_pattern'])
    tfidf_vect.fit(data['tweet'].values.astype('U'))
    xtrain_tfidf = tfidf_vect.transform(train_x.values.astype('U').ravel())
    xvalid_tfidf = tfidf_vect.transform(valid_x.values.astype('U').ravel())
    return [xtrain_tfidf, xvalid_tfidf]

def svm_model(xtrain_tfidf: scipy.sparse.csr.csr_matrix, train_y: pd.Series, xvalid_tfidf: scipy.sparse.csr.csr_matrix, valid_y: pd.Series) -> svm:
    svmm=svm.LinearSVC()
    svmm.fit(xtrain_tfidf,train_y)
    predictions = svmm.predict(xvalid_tfidf)
    score = metrics.f1_score(valid_y, predictions)
    logger = logging.getLogger(__name__)
    logger.info("SVM F1-score: %.5f. ", score)

def rf(xtrain_tfidf: scipy.sparse.csr.csr_matrix, train_y: pd.Series, xvalid_tfidf: scipy.sparse.csr.csr_matrix, valid_y: pd.Series) -> RandomForestClassifier:
    rfc=RandomForestClassifier()
    rfc.fit(xtrain_tfidf,train_y)
    predictions = rfc.predict(xvalid_tfidf)
    score = metrics.f1_score(valid_y, predictions)
    logger = logging.getLogger(__name__)
    logger.info("RF F1-score: %.5f. ", score)

def samp(xtrain_tfidf: scipy.sparse.csr.csr_matrix, train_y: pd.Series) -> scipy.sparse.csr.csr_matrix:
    sm = SMOTE()
    sm_xtrain_tfidf, sm_train_y = sm.fit_resample(xtrain_tfidf, train_y)
    return [sm_xtrain_tfidf, sm_train_y]
