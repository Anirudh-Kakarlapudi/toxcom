import subprocess
import zipfile
import shutil
from pathlib import Path
from toxcom import storage


BERT_CASED = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"


class BertModel:
    """Uses Google-Research's Bert Model to finetune the Bert Model for
    the Toxic Comment Classification Challenge Dataset. The bert-cased
    model is used for this purpose, considering the presence of mixed-case
    words in the comments.

    References:
    1. https://github.com/google-research/bert

    Attributes:
        bert_type (str)
            Type of Bert-Model to use for finetuning.
        max_seq_length (int):
            The maximum total input sequence length after WordPiece
            tokenization. Sequences longer than this will be truncated,
            and sequences shorter than this will be padded.
        train_batch_size (int):
            Total batch size for training.
        learning_rate (float):
            The initial learning rate for Adam.
        num_train_epochs (int):
            Total number of training epochs to perform.
        do_lower_case (bool):
            Whether to lower case the input text. Should be True for
            uncased models and False for cased models.
    """
    def __init__(self, bert_type=BERT_CASED, max_seq_length=400,
                 train_batch_size=8, learning_rate=2e-5,
                 num_train_epochs=3, do_lower_case=True):
        """Initializes :class: ``BertModel``

        Arguments:
            bert_type (str)
                Type of Bert-Model to use for finetuning.
            max_seq_length (int):
                The maximum total input sequence length after WordPiece
                tokenization. Sequences longer than this will be truncated,
                and sequences shorter than this will be padded.
            train_batch_size (int):
                Total batch size for training.
            learning_rate (float):
                The initial learning rate for Adam.
            num_train_epochs (int):
                Total number of training epochs to perform.
            do_lower_case (bool):
                Whether to lower case the input text. Should be True for
                uncased models and False for cased models.
        """
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.do_lower_case = do_lower_case
        self.setup_finetuning(bert_type)


    def setup_finetuning(self, bert_type):
        """Downloads pretrained Bert model files such as 'vocab.txt',
        'bert_config.json', 'bert_model.ckpt', and set up finetuning
        process.

        Arguments:
            bert_type (str)
                Type of Bert-Model to use for finetuning.
        """
        bert_path = storage.get('assets', 'bert')
        bert_model = 'cased_L-12_H-768_A-12'
        if bert_path.exists() and bool(sorted(bert_path.rglob('*'))):
            pass
        else:
            try:
                subprocess.call(f'wget {bert_type}', shell=True)
            except Exception as error:
                print(f'Download Failed: {error}')
            bert_path.mkdir(parents=True)
            with zipfile.ZipFile(f'{bert_model}.zip', 'r') as zip_file:
                zip_file.extractall()
            subprocess.call(f'mv {bert_model}/* {bert_path}/.')
            shutil.rmtree(f'{bert_model}')


    def finetune_bert(self):
        """Calls shell to run google-research's BERT script to finetune the model
        for toxic comment detection dataset.
        """
        run_classifier = storage.get('utils', 'bert', 'run_classifier.py')
        data_dir = storage.get('assets', 'data')
        vocab_file = storage.get('assets', 'bert', 'vocab.txt')
        bert_config_file = storage.get('assets', 'bert', 'bert_config.json')
        init_checkpoint = storage.get('assets', 'bert', 'bert_model.ckpt')
        # Check if output directory exists. Else, create in 'assets' directory.
        output_dir = storage.get('assets', 'bert', 'outputs')
        output_dir.mkdir(parents=True, exist_ok=True)

        subprocess.call(f"""
            python {run_classifier}
            --task_name=cola
            --do_train=True
            --do_predict=True
            --data_dir={data_dir}
            --vocab_file={vocab_file}
            --bert_config_file={bert_config_file}
            --init_checkpoint={init_checkpoint}
            --max_seq_length={self.max_seq_length}
            --train_batch_size={self.train_batch_size}
            --learning_rate={self.learning_rate}
            --num_train_epochs={self.num_train_epochs}
            --do_lower_case={self.do_lower_case}
            --output_dir={output_dir}""", shell=True)
