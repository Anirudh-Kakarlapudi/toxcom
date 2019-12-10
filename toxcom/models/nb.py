"""Contains :class: ``NaiveBayes`` which executes the Naive Bayes model on
the Toxic Comment Classification Challenge to perform multi-label
classification on the six labels in the dataset.
"""
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from toxcom.utils.preprocess import *
import pandas as pd
import re


class NaiveBayes:
    """ A class to perfrorm Naive Bayes
    """

    def __init__(self, use_tf_idf=True, n_gram_range=(1, 1), preprocess=-1):
        """Constructor for the Naive Bayes

        Arguments:
            use_tf_idf(bool):
                Uses `TfIdfVectorizer` if true else uses `CountVectorizer` if false
            n_gram_range(list):
                The vectorizer uses this to build the vocabulary
            preprocess(int):
                Indicates different preprocess techniques represented in `preprocess`
        """
        self.use_tf_idf = use_tf_idf
        self.preprocess = preprocess
        self.n_gram_range = n_gram_range

    def predict(self, train_df, test_df, label_col, target_col):
        """ Trains the NB model for given n_gram_range and predicts the
        all six labels for the test set

        Arguments:
            train_df:(dataframe)
                A train dataframe with 'comment_text' and all labels
            test_df:(dataframe)
                A test dataframe with 'comment_text' and id of comment
            label_col:(list)
                A list of all names of all lavel columns
            target_col:(str)
                A string representing name of text column

        Returns:
            submit_df:(dataframe)
                A dataframe containing the predicted probabilities of all labels
                along with id
        """
        scores = []
        submit_df = pd.DataFrame()
        submit_df['id'] = test_df['id']
        clf = MultinomialNB(alpha=1)

        if self.preprocess != -1:
            clean_dataframe(train_df, self.preprocess, target_col)
            clean_dataframe(test_df, self.preprocess, target_col)
            target_col = 'cleaned_text'
            if self.use_tf_idf:
                vectorizer = TfidfVectorizer(
                    self.n_gram_range,
                    min_df=3,
                    max_df=0.9,
                    use_idf=1,
                    smooth_idf=1,
                    sublinear_tf=1)
            else:
                vectorizer = CountVectorizer(self.n_gram_range)
        else:
            if self.use_tf_idf:
                vectorizer = TfidfVectorizer(
                    strip_accents='ascii',
                    token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',
                    lowercase=True,
                    stop_words='english',
                    ngram_range=self.n_gram_range,
                    min_df=3,
                    max_df=0.9,
                    use_idf=1,
                    smooth_idf=1,
                    sublinear_tf=1)
            else:
                vectorizer = CountVectorizer(
                    strip_accents='ascii',
                    token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',
                    lowercase=True,
                    stop_words='english',
                    ngram_range=self.n_gram_range)

        target_cols = ['toxic', 'severe_toxic', 'obscene',
                       'threat', 'insult', 'identity_hate']
        submit_df = pd.DataFrame()
        submit_df['id'] = test_df['id']

        for target in tqdm(target_cols):
            submit_df[target] = ''
            X_train = vectorizer.fit_transform(train_df['comment_text'])
            X_test = vectorizer.transform(test_df['comment_text'])
            clf = MultinomialNB(alpha=1)
            clf.fit(X_train, train_df[target])
            predictions = clf.predict_proba(X_test)
            prediction = [i[1] for i in predictions]
            submit_df[target] = prediction
        submit_df.to_csv('submit_nb.csv', index=False)
        return submit_df

nb = NaiveBayes(preprocess=-1)
