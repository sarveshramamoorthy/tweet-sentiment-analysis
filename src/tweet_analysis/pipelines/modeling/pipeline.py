from kedro.pipeline import node,Pipeline
from tweet_analysis.pipelines.modeling.nodes import split_data, vector, svm_model, rf, samp


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func = split_data,
                inputs = ['preprocessed_tweets'],
                outputs = ['train_x', 'valid_x', 'train_y', 'valid_y'],
                name = 'splitdata',
            ),
            node(
                func = vector,
                inputs = ['preprocessed_tweets', 'parameters', 'train_x', 'valid_x'],
                outputs = ['xtrain_tfidf', 'xvalid_tfidf'],
                name = 'tfidfvector',
            ),
            node(
                func = svm_model,
                inputs = ['xtrain_tfidf', 'train_y', 'xvalid_tfidf', 'valid_y'],
                outputs = None,
                name = 'svm',
            ),
            node(
                func = rf,
                inputs = ['xtrain_tfidf', 'train_y', 'xvalid_tfidf', 'valid_y'],
                outputs = None,
                name = 'rf',
            ),
            node(
                func = samp,
                inputs = ['xtrain_tfidf', 'train_y'],
                outputs = ['sm_xtrain_tfidf', 'sm_train_y'],
                name = 'smote',
            ),
            node(
                func = svm_model,
                inputs = ['sm_xtrain_tfidf', 'sm_train_y', 'xvalid_tfidf', 'valid_y'],
                outputs = None,
                name = 'sampsvm',
            ),
            node(
                func = rf,
                inputs = ['sm_xtrain_tfidf', 'sm_train_y', 'xvalid_tfidf', 'valid_y'],
                outputs = None,
                name = 'samprf',
            ),
        ]
    )
