import re
import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

porter= PorterStemmer()
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1,pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text,'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat,"", souped)
    clean = stripped
    letters_only = re.sub("[^a-zA-Z]"," ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    stem_sentence = []
    for word in words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    words="".join(stem_sentence).strip()

    return words

def preprocess_tweets(train:pd.DataFrame) -> pd.DataFrame:
    nums = [0,len(train)]
    clean_tweet_texts = []
    for i in range(nums[0], nums[1]):
        clean_tweet_texts.append(tweet_cleaner(train['tweet'][i]))
    train_clean = pd.DataFrame(clean_tweet_texts, columns=['tweet'])
    train_clean['label'] = train.label
    train_clean['id'] = train.id

    return train_clean
