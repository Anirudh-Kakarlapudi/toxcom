import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import defaultdict


class EDA_Toxic:
    """ A class to perform Exploratory Data Analysis.

    Attributes:
        df(dataframe):
            A dataframe with cleaned text and labels
        target_cols:(list)
                Names of label columns
        word_counter:(dict)
            A dictionary counter for the words occuring in dataset
        sentence_counter:(dict)
            A dictionary counter to store the number of sentences per comment
        label_counter:(dict)
            A dictionary counter to store the number of labels per comment
        toxic_counter:(dict)
            A dictionary counter to store the number of clean comments and
            not clean comments
        stop_words_eng:(set)
            A set containing stop words for english language taken from
            nltk corpus
    """

    def __init__(self, df, target_cols):
        """ Initializes :class: ``EDA_Toxic``

        Arguments:
            df:(dataframe)
                A dataframe on whcleaned_textich EDA is to be performed
            target_cols:(list)
                Names of label columns
        """
        self.df = df
        self.target_cols = target_cols
        self.target_df = df[self.target_cols]
        self.word_counter = defaultdict(int)
        self.sentence_counter = defaultdict(int)
        self.label_counter = defaultdict(int)
        self.toxic_counter = defaultdict(int)
        self.stopwords_eng = set(stopwords.words('english'))

    def get_correlation(self):
        """ Function to plot correlation among thetarget columns
        """
        correlation = self.target_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values, annot=True)

    def get_sentences_plot(self):
        """ Function to plot number of sentences for each comment
        """
        self.df['sentences'] = self.df.cleaned_text.apply(
            lambda x: re.split(r'[\!\.\?]', x))
        for sentences in self.df['sentences']:
            self.sentence_counter[len(sentences)] += 1
        sentence_lists = sorted(self.sentence_counter.items())[:20]
        sentence_x, sentence_y = zip(*sentence_lists)
        plt.bar(sentence_x, sentence_y, width=0.5)
        plt.grid()
        plt.xlabel('Number of sentences per comment')
        plt.ylabel('Count of comments')
        plt.title('Sentences per comment vs Counts of comments')
        plt.savefig('sentences.png')

    def word_comparison_plot(self):
        """Creates a violin plot with number of words per each comment
        """
        self.df['word_counts'] = self.df['cleaned_text'].apply(
            lambda x: len(re.split(r'\s+', x)))
        sns.set_style("whitegrid")
        violin = sns.violinplot(
            x='toxic_nature', y='word_counts', data=self.df)
        violin.set_title('Word Comparison Plot')
        violin.set(ylim=(None, 500))
        fig = violin.get_figure()
        fig.savefig("word_comparison.png")

    def create_merged_cols(self):
        """ Creates a merged column with all the labels and also creates a new
        feature with the representation being 'clean' - all labels are zero
        for a comment and 'unclean' - any one of the labels is non-zero
        """
        self.df['merged_labels'] = ''
        self.df['toxic_nature'] = ''
        for index, row in self.df.iterrows():
            list_val = []
            for target in self.target_cols:
                list_val.append(row[target])
                self.df.at[index, 'merged_labels'] = list_val
                count_0 = list_val.count(0)
                count_1 = list_val.count(1)
                self.label_counter[count_1] += 1
                if count_1 == 0:
                    self.df.at[index, 'toxic_nature'] = 'clean'
                    self.toxic_counter['toxic'] += 1
                else:
                    self.df.at[index, 'toxic_nature'] = 'not clean'
                    self.toxic_counter['not toxic'] += 1

    def get_label_plots(self):
        """ Plots and saves the comments vs label plot and toxic nature of
        comments plot
        """
        toxic_lists = sorted(self.toxic_counter.items())
        toxic_x, toxic_y = zip(*toxic_lists)
        label_lists = sorted(self.label_counter.items())
        label_x, label_y = zip(*label_lists)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.grid()
        plt.bar(toxic_x, toxic_y, width=0.5)
        plt.xlabel('number of labels in all comments')
        plt.ylabel('number of comments')
        plt.title('comments vs labels')
        plt.subplot(132)
        plt.plot(label_x, label_y)
        plt.grid()
        plt.title('Toxic Nature of comments')
        plt.xlabel('number of labels per comment')
        plt.ylabel('number of comments')
        plt.savefig('label_plots.png')
        plt.show()

    def get_top_words_plot(self, num_words=15, stopwords=True):
        """ Plots the top num_words with frequency counts

        Arguments:
            stopwords:(bool):
                If True -Removes the stop words from the counter and
                plots the top num_words remaining words
            num_words:(int)
                Number of words with frequency counts that are to be plotted
        """
        self.df['words'] = self.df.cleaned_text.str.strip().str.split('[\W_]+')
        for words in self.df['words']:
            for word in words:
                if stopwords:
                    if word not in self.stopwords_eng:
                        self.word_counter[word] += 1
                else:
                    self.word_counter[word] += 1
        sorted_dict = sorted(self.word_counter.items(),
                             key=lambda x: x[1], reverse=True)
        top_n_words = sorted_dict[:num_words]
        words, counts = zip(*top_n_words)
        plt.rcParams["figure.figsize"] = (15, 5)
        plt.grid()
        plt.xlabel('word')
        plt.ylabel('counts')
        plt.title('Top words with highest frequency counts')
        plt.plot(words, counts)
        plt.savefig('top_words.png')
