"""Contains :class: ``BertModel`` which executes the bert model on 
the Toxic Comment Classification Challenge to perform multi-label
classification on the six labels in the dataset.

This code is based on the ``run_classifier.py`` script in [1].

References:
-----------
[1] https://github.com/google-research/bert
"""

import json
import subprocess
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import tensorflow as tf
import bert
from bert import optimization
from bert import tokenization
from bert import modeling
from toxcom import storage


BERT_CASED = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"


class InputExample:
    """A single train/test example for simple sequence classification.

    Attributes:
        identifier (str):
            Unique ID for the train/test example.
        text (str):
            Untokenized text.
        labels (str):
            Label of the example
    """
    def __init__(self, guid, text, labels=None):
        """Constructs an InputExample.

        Arguments:
            guid (str):
                Unique ID for the train/test example.
            text (str):
                Untokenized text.
            labels (str):
                Label of the example
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures:
    """A single set of features of data.

    Attributes:
        input_ids ():
            List of list of input ids for features.
        input_mask ():
            List of list of input masks for features.
        segment_ids ():
            List of list of segment ids for features.
        label_ids (list):
            List of list of label ids for features.
        is_real_example ():
            True, if real example, else, False.
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
        """Constructs an InputFeature.

        Arguments:
            input_ids ():
                List of list of input ids for features.
            input_mask ():
                List of list of input masks for features.
            segment_ids ():
                List of list of segment ids for features.
            label_ids (list):
                List of list of label ids for features.
            is_real_example ():
                True, if real example, else, False.
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class BertModel:
    """Uses Google-Research's Bert Model to finetune the Bert Model for
    the Toxic Comment Classification Challenge Dataset. The bert-cased
    model is used for this purpose, considering the presence of mixed-case
    words in the comments.

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
        warmup_proportion (float):
            Proportion of the number of training steps to include as
            warmup steps.
        save_checkpoints_steps (int):
            Number of checkpoints steps to save.
        save_summary_steps (int):
            Number of summary steps to save.
        do_lower_case (bool):
            Whether to lower case the input text. Should be True for
            uncased models and False for cased models.
    """
    def __init__(self, bert_type=BERT_CASED, **kwargs):
        """Initializes :class: ``BertModel``

        Arguments:
            bert_type (str)
                Type of Bert-Model to use for finetuning.
        """
        # Default Parameters
        params = {
            'max_seq_length': 400,
            'train_batch_size': 8,
            'learning_rate': 2e-5,
            'num_train_epochs': 3,
            'warmup_proportion': 0.1,
            'save_checkpoints_steps': 1000,
            'save_summary_steps': 500,
            'do_lower_case': True,
        }

        for param in params:
            if param in kwargs and kwargs[param]:
                params[param] = kwargs[param]

        # Model Parameters
        self.max_seq_length = params['max_seq_length']
        self.train_batch_size = params['train_batch_size']
        self.learning_rate = params['learning_rate']
        self.num_train_epochs = params['num_train_epochs']
        self.warmup_proportion = params['warmup_proportion']
        self.save_checkpoints_steps = params['save_checkpoints_steps']
        self.save_summary_steps = params['save_summary_steps']
        self.do_lower_case = params['do_lower_case']
        self.bert_model = 'cased_L-12_H-768_A-12'
        # Downloads and sets up bert's assets.
        self.setup_finetuning(bert_type)
        # Retrives absolute path of ``assets``
        self.vocab_file = storage.get('toxcom', 'assets', 'bert', self.bert_model, 'vocab.txt')
        self.config = storage.get('toxcom', 'assets', 'bert', self.bert_model, 'bert_config.json')
        self.init_checkpoint = storage.get('toxcom', 'assets', 'bert', self.bert_model, 'bert_model.ckpt')
        # Check if output directory exists. Else, create in ``assets`` directory.
        self.output_dir = storage.get('toxcom', 'assets', 'bert', 'outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def setup_finetuning(self, bert_type):
        """Downloads pretrained Bert model files such as 'vocab.txt',
        'bert_config.json', 'bert_model.ckpt', and set up finetuning
        process.

        Arguments:
            bert_type (str)
                Type of Bert-Model to use for finetuning.
        """
        bert_path = storage.get('toxcom', 'assets', 'bert')

        if bert_path.exists() and bool(sorted(bert_path.rglob('*'))):
            pass
        else:
            try:
                subprocess.call(f'wget {bert_type}', shell=True)
            except Exception as error:
                print(f'Download Failed: {error}')
            bert_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(f'{self.bert_model}.zip', 'r') as zip_file:
                zip_file.extractall(bert_path)
            subprocess.call(f"rm {self.bert_model}*",shell=True)


    def create_examples(self, df, labels_available=True):
        """Creates list of :class: ``InputExample`` for train/dev/test :class:
        ``pandas.DataFrame``.

        Arguments:
            df (pandas.DataFrame):
                DataFrame for train/dev/test set.
            labels_available (True):
                Default is ``True``.

        Returns:
            examples (list):
                List of InputExamples.
        """
        examples = []

        for row in df.values:
            identifier = row[0]
            text = row[1]
            if labels_available:
                labels = row[2:]
            else:
                labels = [0, 0, 0, 0, 0, 0]
            examples.append(InputExample(guid=identifier,
                                         text=text,
                                         labels=labels))
        return examples


    def convert_examples_to_features(self, examples, tokenizer):
        """Converts list of :class: ``InputExample`` to a list of
        :class: ``InputFeature``.

        Arguments:
            examples (list):
                List of InputExamples.
            tokenizer:
                BertTokenizer.

        Returns:
            features (list):
                List of InputFeatures.
        """
        features = []
        for index, example in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text)
            # Modifies 'tokens_a' in place such that the total length is less
            # than the max_seq_length.
            # Account for [CLS] and [SEP] with "-2".
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
            # For classification tasks, the first vector (corresponding to
            # [CLS]) is used as the sentence vector. Note that this only
            # makes sense because the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has '1' for real tokens and '0' for padding tokens. Only
            # real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero pad up to the sequence length
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            labels_ids = [int(label) for label in example.labels]

            if index < 0:
                logger.info("*** Example ***")
                logger.info(f"ID: {example.guid}")
                logger.info(f"Tokens: {' '.join([str(x) for x in tokens])} ")
                logger.info(f"Input_ids: {' '.join([str(x) for x in input_ids])} ")
                logger.info(f"Input_mask: {' '.join([str(x) for x in input_mask])} ")
                logger.info(f"Segment_ids: {' '.join([str(x) for x in segment_ids])} ")
                logger.info(f"Labels: {' '.join([str(x) for x in labels_ids])} ")

            features.append(
                InputFeatures(input_ids =input_ids, input_mask=input_mask,
                              segment_ids=segment_ids, label_ids=labels_ids))
            return features


    def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model.

        Arguments:
            is_training (bool):
                True, if training; False otherwise.
            input_ids ():
                List of list of input ids for features.
            input_mask ():
                List of list of input masks for features.
            segment_ids ():
                List of list of segment ids for features.
            labels (list):
                List of list of label ids for features.
            num_labels (int):
                Number of labels in the output layer.
            use_one_hot_embeddings (bool):
                To whether use one-hot embeddings for the outputs in the model.

        Returns:
            loss, per_example_loss, logits, probabilities (tuple):
                Model parameters for training, returned only if ``is_training``
                is True.
        """
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.sigmoid(logits)

            labels = tf.cast(labels, tf.float32)

            tf.logging.info("Number of labels: {num_labels}", "Logits: {logits}, Labels: {labels}")
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)


    def model_fn_builder(self, bert_config, num_labels, num_train_steps, num_warmup_steps,
                         use_one_hot_embeddings):
        """Builds :class: ``tensorflow.model_fn_builder``.

        Arguments:
            bert_config (dict):
                Configuration for bert model
            num_labels (int):
                Number of labels in the output layer.               
            num_train_steps (int):
                Number of training steps.
            num_warmup_steps (int):
                Number of warmup steps.
            use_one_hot_embeddings (bool):
                To whether use one-hot embeddings for the outputs in the model.

        Returns:
            model_fn:
                Tensorflow EstimatorSpec
        """
        def model_fn(features, labels, mode, params):
            """Builds a :class: tensorflow.estimator.EstimatorSpec for the
            given mode, loss and evaluation metrics.

            Arguments:
                features (list):
                    List of features in the dataset.
                labels (list):
                    List of label ids for the features.
                params (dict):
                    Dictionary of parameters.

            Returns:
                output_spec (tensorflow.estimator.EstimatorSpec)
                    EstimatorSpec for the given mode, loss and evaluation
                    metrics.
            """
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_real_example = None

            if "is_real_example" in features:
                is_real_example = tf.cast(features["is_real_example"],
                                          dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids),
                                          dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = self.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}

            if self.init_checkpoint:
                (assignment_map,
                 initialized_variable_names
                ) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                str(self.init_checkpoint))

                tf.train.init_from_checkpoint(str(self.init_checkpoint),
                                              assignment_map)

            tf.logging.info("***** Training Variables *****")

            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, self.learning_rate, num_train_steps, num_warmup_steps, False)

                output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                         loss=total_loss,
                                                         train_op=train_op,
                                                         scaffold=None)

            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):
                    """Returns a dictionary of evaluation metrics.

                    Arguments:
                        per_example_loss (float):
                            Loss per example in the list of InputExamples.
                        label_ids (list):
                            List of label id's for an example in the list of
                            InputExamples.
                        probabilites (list):
                            Probabilities for output nodes in an iteration.
                        is_real_example (bool):
                            Whether a real example or not.

                    Returns:
                        eval_dict (dict):
                            Dictionary of evaluation metrics.
                    """
                    logits_split = tf.split(probabilities, num_labels, axis=-1)
                    label_ids_split = tf.split(label_ids, num_labels, axis=-1)

                    eval_dict = {}
                    for j, logits in enumerate(logits_split):
                        label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                        current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                        eval_dict[str(j)] = (current_auc, update_op_auc)
                    eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                    return eval_dict

                eval_metrics = metric_fn(per_example_loss, label_ids, probabilities,
                                         is_real_example)
                output_spec = tf.estimator.EstimatorSpec(
                                                mode=mode,
                                                loss=total_loss,
                                                eval_metric_ops=eval_metrics,
                                                scaffold=None)

            else:
                print("mode:", mode, "probabilities:", probabilities)
                output_spec = tf.estimator.EstimatorSpec(
                                                mode=mode,
                                                predictions={"probabilities:", probabilities},
                                                scaffold=None)
            return output_spec
        return model_fn


    def input_fn_builder(self, features, is_training):
        """Creates an `input_fn` closure to be passed to TPUEstimator.

        Arguments:
            features (list):
                List of InputFeatures generated from list of InputExamples.
            is_training (bool):
                True, if training step, else False.

        Returns:
            input_fn (func):
                Function which builds the dataset.
        """
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids = [], [], [], []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)

        def input_fn(params):
            """The actual input function.
            
            Arguments:
                params (dict):
                    Input Parameters.

            Returns:
                dataset (tensorflow.data.Dataset):
                    Dataset built from the data source.
            """
            batch_size = params["batch_size"]
            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            dataset = tf.data.Dataset.from_tensor_slices({
                      "input_ids":
                            tf.constant(
                                all_input_ids,
                                shape=[num_examples, self.max_seq_length],
                                dtype=tf.int32),
                      "input_mask":
                            tf.constant(
                                all_input_mask,
                                shape=[num_examples, self.max_seq_length],
                                dtype=tf.int32),
                      "segment_ids":
                            tf.constant(
                                all_segment_ids,
                                shape=[num_examples, self.max_seq_length],
                                dtype=tf.int32),
                      "label_ids":
                            tf.constant(
                                all_label_ids,
                                shape=[num_examples, 6],
                                dtype=tf.int32),
            })

            if is_training:
                dataset = dataset.repeat()
                dataset = dataset.shuffle(buffer_size=100)

            dataset = dataset.batch(batch_size=batch_size,
                                    drop_remainder=False)
            return dataset
        return input_fn


    def execute(self, train_df, eval_df):
        """Trains and evaluates the performance of :class: ``pandas.DataFrame``
        for the :class: ``BertModel``, sent as arguments to the function.

        Arguments:
            train_df (pandas.DataFrame):
                Pandas DataFrame for the train set.
            eval_df (pandas.DataFrame):
                Pandas DataFrame for the evaluation set.
        """
        tokenization.validate_case_matches_checkpoint(True, 'bert_model.ckpt')
        tokenizer = tokenization.FullTokenizer(
            vocab_file=str(self.vocab_file), do_lower_case=self.do_lower_case)

        run_config = tf.estimator.RunConfig(
            model_dir=self.output_dir,
            save_summary_steps=self.save_summary_steps,
            keep_checkpoint_max=1,
            save_checkpoints_steps=self.save_checkpoints_steps)

        train_examples = self.create_examples(train_df)

        num_train_steps = int(len(train_examples) / self.train_batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        train_features = self.convert_examples_to_features(train_examples, tokenizer)        

        train_input_fn = self.input_fn_builder(features=train_features, is_training=True)
        bert_config = modeling.BertConfig.from_json_file(str(self.config))
        model_fn = self.model_fn_builder(
            bert_config=bert_config,
            num_labels=6,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_one_hot_embeddings=False)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": self.train_batch_size})

        print('Beginning Training!')
        train_begin_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('Training took time: ', datetime.now() - train_begin_time)

        eval_examples = self.create_examples(eval_df)
        eval_features = self.convert_examples_to_features(eval_examples, tokenizer)

        eval_input_fn = self.input_fn_builder(features=eval_features, is_training=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)
        print("Evaluation Results: ", result)
        print("Writing results to assets/results_bert.json")

        output_file = storage.get('toxcom', 'assets', 'results_bert.json')
        with open(output_file, 'w') as write_results:
            json.dump(result, write_results)
