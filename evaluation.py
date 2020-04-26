from collections import Counter
from keras.callbacks import Callback
from conf import Config as cf
import pickle
from parse_squad_data import SquadExample, Answer
import os
import pandas as pd
import numpy as np
import re
import string
from exponential_moving_average import ExponentialMovingAverage

class EvaluationCallback(Callback):
    """
    This callback is in charge of evaluating on the validation data at the end of each epoch, as well as within-epoch
    validations every defined number of training steps (on a subset of the validation data).
    It also maintains the ExponentialMovingAverage of the weights (see attention-question-answering paper, section 4.1.1)
    """
    def __init__(self):
        super(EvaluationCallback, self).__init__()
        self.global_step = 0
        self.keras_ema = None
        self.tokenizer = pickle.load(open(cf.FIT_TOKENIZER_PATH, 'rb'))
        self.word_index = self.tokenizer.word_index
        self.within_epoch_metrics = []
        self._load_data(cf.PARSED_VAL_DATA_PATH)

    def on_train_begin(self, logs=None):
        if cf.APPLY_EMA:
            self.keras_ema = ExponentialMovingAverage(model=self.model, decay=cf.EMA_DECAY,
                                                  temp_model_pth='./logs/temp_model.h5')

    def _load_data(self, path):
        if cf.LIMIT_VAL is not None:
            self.data = pickle.load(open(path, 'rb'))[:cf.LIMIT_VAL]
        else:
            self.data = pickle.load(open(path, 'rb'))
        if cf.SHUFFLE_DATA:
            np.random.shuffle(self.data)
        self.val_tokenized_contexts = np.array([e.context_tokenized for e in self.data])
        self.val_tokenized_questions = np.array([e.question_tokenized for e in self.data])
        self.y_spans = [] # List of lists of spans per answer, per example
        self.y_texts = []
        for example in self.data:
            tok_spans = []
            text_answers = []
            for ans in example.answers:
                tok_spans.append(ans.tokenized_span)
                text_answers.append(self.tokenizer.sequences_to_texts([ans.tokenized])[0]) # convert the tokenized ansewr back to text. Note this is not the same as ans.text

            self.y_texts.append(text_answers)
            self.y_spans.append(tok_spans)

    def _get_predictions(self, limit=None):
        # Allow predicting just a part of the validation set:
        if limit is not None:
            x_input = [self.val_tokenized_contexts[:limit,:], self.val_tokenized_questions[:limit,:]]
        else:
            x_input = [self.val_tokenized_contexts, self.val_tokenized_questions]

        _, _, predicted_starts, predicted_ends = self.model.predict(x=x_input, batch_size=cf.BATCH_SIZE)
        predicted_starts = [int(pos) for pos in predicted_starts]
        predicted_ends = [int(pos) for pos in predicted_ends]
        predictes_spans = list(zip(predicted_starts, predicted_ends))
        return predictes_spans

    def _predicted_spans_to_texts(self, predicted_spans):
        predicted_texts = []
        for predicted_span, example in zip(predicted_spans, self.data):
            s, e = predicted_span
            predicted_tokenized_answer = example.context_tokenized[s:e+1]
            predicted_text_answer = self.tokenizer.sequences_to_texts([predicted_tokenized_answer])[0]
            predicted_texts.append(predicted_text_answer)
        return predicted_texts

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1 # keep count of global batch number (unlike batch param which is within the current epoch)
        if cf.APPLY_EMA:
            self.keras_ema.average_update()
        if cf.EVALUATE_WITHIN_EPOCH_EVERY_N_STEPS is not None and \
            self.global_step % cf.EVALUATE_WITHIN_EPOCH_EVERY_N_STEPS == 0:
            predicted_spans = self._get_predictions(limit=cf.WITHIN_EPOCH_EVALUATION_LIMIT)
            predicted_texts = self._predicted_spans_to_texts(predicted_spans)
            f1, em = self.evaluate(predicted_texts)
            self.within_epoch_metrics.append({'step':self.global_step, 'f1':f1,'em':em, 'loss':logs['loss']})
            print("\nStep #{}  -- validation metrics ({} val examples) : f1={}  em={}".format(
                self.global_step,cf.WITHIN_EPOCH_EVALUATION_LIMIT, f1, em ))

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if not cf.VALIDATE_ON_EPOCH_END:
            return
        if cf.APPLY_EMA:
            self.keras_ema.assign_shadow_weights(backup=True)
        self.model.save_weights(cf.MODEL_CHECKPOINT_FILE.format(epoch=epoch_num))
        predicted_spans = self._get_predictions()
        predicted_texts = self._predicted_spans_to_texts(predicted_spans)
        # Dump predictions to file
        preds_file_path = os.path.join(cf.VAL_PREDICTIONS_DIR, "epoch_{}_val_preds.csv".format(epoch_num))
        preds_df = pd.DataFrame(predicted_spans, columns=['pred_start', 'pred_end'])
        preds_df.to_csv(preds_file_path)

        # Evaluate:
        f1, em = self.evaluate(predicted_texts)
        print("Epoch: {}, Validation Metrics: F1={} EM={}".format(epoch_num, f1, em))
        self._write_metrics(epoch_num=epoch_num, metric_names=['F1', 'EM', 'loss'], metric_values=[f1, em, logs['loss']])
        # if within-epoch evaluation was done, write these metrics as well:
        if cf.EVALUATE_WITHIN_EPOCH_EVERY_N_STEPS:
            self._write_within_epoch_metrics()
        if cf.APPLY_EMA:
            print("Loading back weights without EMA")
            self.model.load_weights(self.keras_ema.temp_model_pth)

    @staticmethod
    def _write_metrics(epoch_num, metric_names, metric_values):
        row = {metric_name:metric_value for metric_name, metric_value in zip(metric_names, metric_values)}
        row['epoch'] = epoch_num
        if epoch_num == 1 or not os.path.exists(cf.METRICS_LOGS_FILE):
            df = pd.DataFrame([row])
        else:
            df = pd.read_csv(cf.METRICS_LOGS_FILE, index_col=0)
            df = df.append([row])
        df.to_csv(cf.METRICS_LOGS_FILE)

    def _write_within_epoch_metrics(self):
        df = pd.DataFrame(self.within_epoch_metrics)
        df.to_csv(cf.WITHIN_EPOCH_METRICS_FILE)

    def _get_max_f1(self, predicted_text, y_texts):
        max_f1 = max([self._get_f1(a_gold=y_text, a_pred=predicted_text) for y_text in y_texts])
        return max_f1

    def _get_f1(self, a_gold, a_pred):
        """
        Calculate F1 in text-space, with some pre-cleaning of the texts. See SquadEvaluator for more details.
        """
        return SquadEvaluator.compute_f1(a_gold, a_pred)

    def _get_max_em(self, predicted_text, y_texts):
        max_em = max([self._get_em(a_gold=y_text, a_pred=predicted_text) for y_text in y_texts])
        return max_em

    def _get_em(self, a_gold, a_pred):
        """
        Calculate EM in text-space, with some pre-cleaning of the texts. See SquadEvaluator for more details.
        """
        return SquadEvaluator.compute_em(a_gold, a_pred)

    def evaluate(self, predicted_texts):
        f1 = np.mean([self._get_max_f1(predicted_text, y_texts) for predicted_text, y_texts \
                      in zip(predicted_texts, self.y_texts)])
        em = np.mean([self._get_max_em(predicted_text, y_texts) for predicted_text, y_texts \
                      in zip(predicted_texts, self.y_texts)])
        return f1, em


class SquadEvaluator(object):
    """
    All methods below are from the official SQuAD 2.0 eval script
    https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """
    @staticmethod
    def normalize_answer(s):
        """Convert to lowercase and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return SquadEvaluator.normalize_answer(s).split()

    @staticmethod
    def compute_em(a_gold, a_pred):
        return int(SquadEvaluator.normalize_answer(a_gold) == SquadEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = SquadEvaluator.get_tokens(a_gold)
        pred_toks = SquadEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
