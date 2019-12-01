"""
Entry-point into :package: ``toxcom``.
To get the list of options, execute:

    $ python -m run_experiments --help
"""


import argparse
import pandas as pd
from toxcom import storage
from toxcom.utils import preprocess
from toxcom.models.transformers import *


def read_data():
    """Reads data from toxcom/assets/data to return train and eval dataframes.

    Returns:
        train_df, eval_df (pandas.DataFrame, pandas.DataFrame)
            Train and Evaluation pandas DataFrames.
    """
    data_dir = storage.get('toxcom', 'assets', 'data')
    train_data_path = Path(data_dir) / 'train.csv'
    test_data_path = Path(data_dir) / 'test.csv'
    test_labels_path = Path(data_dir) / 'test_labels.csv'

    train_df = pd.read_csv(train_data_path)
    eval_df = pd.read_csv(test_data_path)
    eval_labels_df = pd.read_csv(test_labels_path)
    pd.merge(eval_df, eval_labels_df, how='left', on='id')

    return train_df, eval_df


def run_bert(train_df, eval_df, **kwargs):
    """Runs :class: ``toxcom.transformers.BertModel``
    and writes results to toxcom/assets/results_bert.json.

    Arguments:
        train_df (pandas.DataFrame)
            Pandas DataFrame for train-set.
        eval_df (pandas.DataFrame)
            Pandas DataFrame for eval-set.
    """
    model = BertModel(bert_type=BERT_CASED, **kwargs)
    model.execute(train_df, eval_df)


def main():
    """Main function
    """
    parser = argparse.ArgumentParser(
        description='Toxic Comment Classification Challenge')

    # BERT Model Arguments
    parser.add_argument('--max_seq_length', dest='max_seq_length', type=int,
                        help='Maximum Sequence Length.')
    parser.add_argument('--train_batch_size', dest='train_batch_size',
                        type=int, help='Batch size for training.')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        help='Learning rate.')
    parser.add_argument('--num_train_epochs', dest='num_train_epochs', type=int,
                        help='Number of epochs for training.')
    parser.add_argument('--warmup_proportion', dest='warmup_proportion',
                        type=float, help='Percentage of training steps for warmup.')
    parser.add_argument('--save_checkpoints_steps', dest='save_checkpoints_steps',
                        type=int, help='Number of checkpoint steps to save.')
    parser.add_argument('--save_summary_steps', dest='save_summary_steps',
                        type=int, help='Number of summary steps to save.')
    parser.add_argument('--do_lower_case', dest='do_lower_case', type=bool,
                        help='Whether to convert to lower case or not.')

    args = parser.parse_args()

    kwargs = {}
    for arg in vars(args):
        kwargs[arg] = getattr(args, arg)

    # Running experiments on :class: ``toxcom.models.BertModel``
    train_df, eval_df = read_data()
    run_bert(train_df, eval_df, **kwargs)


if __name__ == '__main__':
    main()
