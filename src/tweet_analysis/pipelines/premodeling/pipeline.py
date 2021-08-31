from kedro.pipeline import node,Pipeline
from tweet_analysis.pipelines.premodeling.nodes import preprocess_tweets


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func = preprocess_tweets,
                inputs = 'train',
                outputs = 'preprocessed_tweets',
                name = 'preprocessing_tweets',
            )
        ]
    )
