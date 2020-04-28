from model import build_model
from conf import Config as cf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--weights", default=r'./trained_models/128d_13epochs.hdf5', type=str)
    args = arg_parser.parse_args()

    model = build_model(cf)
    model.load_weights(args.weights)

    tokenizer = pickle.load(open(cf.FIT_TOKENIZER_PATH, 'rb'))
    oov_token = tokenizer.oov_token
    word2idx = tokenizer.word_index
    idx2word = {v:k for k,v in word2idx.items()}

    context = ''
    while context != 'exit':
        print("Enter the context information (or 'exit'):")
        context = input()
        if context == 'exit':
            continue

        another_question = 'y'
        while another_question in ['yes', 'y','Y']:
            print("Enter the question:")
            question = input()
            t_context, t_question = tokenizer.texts_to_sequences([context, question])

            t_context = pad_sequences([t_context], maxlen=cf.MAX_CONTEXT_LENGTH)[0]
            t_question = pad_sequences([t_question], maxlen=cf.MAX_QUESTION_LENGTH)[0]

            _, _, start_idx, end_idx = model.predict(x=[np.array([t_context]), np.array([t_question])])
            start_idx, end_idx = int(start_idx[0]), int(end_idx[0])

            predicted_indices = t_context[start_idx:end_idx + 1]
            predicted_tokens = [idx2word[idx] for idx in predicted_indices]
            predicted_tokens = [tok if tok != oov_token else '<UNK>' for tok in predicted_tokens]
            predicted_answer = " ".join(predicted_tokens)

            print("Answer: ", predicted_answer)
            print("Another question for same context? (y/n)")
            another_question = input()