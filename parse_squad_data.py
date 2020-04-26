from conf import Config as cf
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
from pprint import pprint
from collections import Counter
import argparse

int_to_word_map = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six', '7':'seven','8':'eight','9':'nine'}


class ANSWER_STATUS:
    found_once = 0
    used_closest_occurrence_logic = 1
    used_first_occurrence = 2
    not_found = 3

    @staticmethod
    def code_to_str(code):
        if code == ANSWER_STATUS.found_once:
            return 'found once'
        if code == ANSWER_STATUS.used_closest_occurrence_logic:
            return 'used closest occurence logic'
        if code == ANSWER_STATUS.used_first_occurrence:
            return 'used first occurence'
        if code == ANSWER_STATUS.not_found:
            return 'not found'


class Answer(object):
    def __init__(self, answer_dict, tokenizer):
        self.start_pos = answer_dict.get("answer_start")
        self.text = answer_dict.get("text", '').lower()
        self.end_pos = self.start_pos + len(self.text)
        self.tokenized = tokenizer.texts_to_sequences([self.text])[0]
        self.answer_status = None

    def set_tokenized_span(self, context, context_tokenized):
        self.tokenized_span = self.get_tokenized_span(context, context_tokenized)

    def get_tokenized_span(self, context, context_tokenized):
        """
        :return: an approximate span of the answer in the tokenized context. Approximate becase the original span in char-
                based, and there some some discrepancy after tokenization (for example punctuation).
                Another source of error here is answers that are mapped to UNK tokens.
        """
        tokenized_c = list(context_tokenized)
        tokenized_a = self.tokenized
        answer = self.text
        answer_start_char = self.start_pos

        spans = [(i, i + len(tokenized_a) - 1) for i in range(len(tokenized_c)) if
                 tokenized_c[i:i + len(tokenized_a)] == tokenized_a]
        if len(spans) == 1:  # found just once
            self.answer_status = ANSWER_STATUS.found_once
            return spans[0]

        if len(spans) == 0: # answer NOT found in context
            # A hack to deal with integer answers:
            if len(answer) == 1:
                word_answer = int_to_word_map.get(answer)

                if word_answer is not None:
                    tokenized_word_answer = tokenizer.texts_to_sequences([word_answer])[0]
                    spans = [(i, i + len(tokenized_word_answer) - 1) for i in range(len(tokenized_c)) if
                             tokenized_c[i:i + len(tokenized_word_answer)] == tokenized_word_answer]

            if len(spans) == 0:
                self.answer_status = ANSWER_STATUS.not_found
                return ANSWER_STATUS.not_found

        # answer appears more than once in context:
        positions = [m.start() for m in re.finditer(re.escape(answer), context)]
        diff = [np.abs(p - answer_start_char) for p in positions]
        closest = int(np.argmin(diff))
        try:
            self.answer_status = ANSWER_STATUS.used_closest_occurrence_logic
            return spans[closest]
        except IndexError:
            self.answer_status = ANSWER_STATUS.used_first_occurrence
            return spans[0] # Just take the first, this happens rarely


class SquadExample(object):
    def __init__(self, _id, context, question, answers, tokenizer):
        self.id = _id
        self.context = context.lower()
        self.question = question.lower()
        self.answers = [Answer(ans, tokenizer) for ans in answers]

        self.context_tokenized = pad_sequences(tokenizer.texts_to_sequences([context]), maxlen=cf.MAX_CONTEXT_LENGTH)[0]
        self.question_tokenized = pad_sequences(tokenizer.texts_to_sequences([self.question]), maxlen=cf.MAX_QUESTION_LENGTH)[0]

        # For each ans, add it's span in the tokenized space:
        for answer in self.answers:
            answer.set_tokenized_span(self.context, self.context_tokenized)

        best_answer_idx = np.argmin([ans.answer_status for ans in self.answers]) # Min status code is the best
        self.gt = self.answers[best_answer_idx]


def context_qas_iterator(data_path):
    raw_data = json.load(open(data_path, 'r')).get('data')
    for topic in raw_data:
        paragraphs = topic.get('paragraphs')
        for para in paragraphs:
            context = para.get('context')
            for qa_dict in para.get('qas'):
                question = qa_dict.get('question')
                answers = qa_dict.get('answers', [])
                _id = qa_dict.get('id','')
                yield _id, context, question, answers


def fit_tokenizer():
    """
    :return: a keras Tokenizer object, fit on the squad data.
    """
    itr = context_qas_iterator(cf.SQUAD_TRAIN_DATA_PATH)
    all_contexts, all_questions = [], []
    for _id, context, question, answers in itr:
        all_contexts.append(context)
        all_questions.append(question)

    itr = context_qas_iterator(cf.SQUAD_VAL_DATA_PATH)
    for _id, context, question, answers in itr:
        all_contexts.append(context)
        all_questions.append(question)

    # Tokenize text corpus:
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(all_contexts+all_questions)
    return tokenizer

def parse_squad_data(path, tokenizer):
    def _is_valid_answer(ans):
        answer_found = ans.answer_status != ANSWER_STATUS.not_found
        if not answer_found:
            return False
        y_start, y_end = ans.tokenized_span
        valid_span = y_end >= y_start

        return valid_span

    parsed_examples = []

    itr = context_qas_iterator(path)
    for _id, context, question, answers in itr:

        parsed_example = SquadExample(_id, context, question, answers, tokenizer)
        # Keep only examples where we successfully found the answer span in the context:
        if parsed_example.gt.answer_status == ANSWER_STATUS.not_found:
            continue
        # Remove other invalid answers:
        parsed_example.answers = list(filter(lambda x: _is_valid_answer(x),
                                             parsed_example.answers))
        parsed_examples.append(parsed_example)

    return parsed_examples

def get_embeddings_for_vocab(path, vocab_word_index, print_every=None):
    """
    Load GLoVe data do a hash map, only for words that are present in vocab_word_index.
    This avoids ever loading the full embedding matrix to memory.
    """
    wv = {}
    line_counter = 0
    with open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            word = line[:line.find(" ")]
            if word in vocab_word_index:
                tokens = line.strip().split(' ')
                wv[word] = [float(t) for t in tokens[1:]]
            line_counter += 1
            if print_every is not None and line_counter % print_every == 0:
                print("\t\t--processed {} lines.".format(line_counter))
    return wv



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--fit_tokenizer", default='true', type=str)
    arg_parser.add_argument("--create_reduced_embeddings", default='true', type=str)

    args = arg_parser.parse_args()

    if args.fit_tokenizer == 'true':
        print("Fitting tokenizer..")
        tokenizer = fit_tokenizer()
        with open("./data/squad/fit_tokenizer.pickle",'wb') as f:
            pickle.dump(tokenizer, f)
    else:
        print("Loading tokenizer from pickle..")
        tokenizer = pickle.load(open(cf.FIT_TOKENIZER_PATH,'rb'))

    print("\tDone.")
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(cf.VOCAB_SIZE, len(word_index) + 1)


    print("\nParsing validation data...")
    validation_data = parse_squad_data(cf.SQUAD_VAL_DATA_PATH, tokenizer)
    pickle.dump(validation_data, open(cf.PARSED_VAL_DATA_PATH, 'wb'))
    print("\tDone (parsed {} validation examples).".format(len(validation_data)))

    print("\nParsing train data...")
    train_data = parse_squad_data(cf.SQUAD_TRAIN_DATA_PATH, tokenizer)
    pickle.dump(train_data, open(cf.PARSED_TRAIN_DATA_PATH, 'wb'))
    print("\tDone (parsed {} train examples).".format(len(train_data)))


    if args.create_reduced_embeddings == 'true':
        print("\nCreating reduced embedding matrix, only with words seen in corpus")
        print("\tLoading embeddings for vocabulary found in data...")
        embeddings_index = get_embeddings_for_vocab(cf.PRETRAINED_EMBEDDINGS_PATH, word_index, print_every=10**5)
        print("\t\tDone.")
        unk_embedding = np.random.normal(loc=0, scale=1,size=cf.EMBEDDING_DIM)
        numwords = len(list(word_index.items()))+1
        reduced_embeddings = np.zeros((numwords, cf.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will as initialized
                reduced_embeddings[i] = embedding_vector
            else:
                reduced_embeddings[i] = unk_embedding
        np.save(cf.REDUCED_EMBEDDING_MATRIX_PATH, reduced_embeddings)
        print("Done.")

    # Print answer parsing stats:
    print("\nAnswer Statistics:")
    pprint(Counter([ANSWER_STATUS.code_to_str(example.gt.answer_status) for example in train_data+validation_data]))

