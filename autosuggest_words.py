import numpy as np
import nltk
from collections import defaultdict
import pandas as pd


#This function will return the Count matrix along witht the bigrams and list of vocabulary
def get_count_matrix(data , limit):

    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0][:limit]

    tokenized_sent = []
    stop = ['.','...',',',';',')','(','!']
    for sent in sentences:
        sentence = sent.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized = [w for w in tokenized if w not in stop]
        tokenized_sent.append(tokenized)

    bigrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(dict)

    for sent in tokenized_sent:
        m = len(sent)
        for i in range(m-3+1):
            trigram = tuple(sent[i : i + 3])
            bigram = trigram[0 : -1]

            if not bigram in bigrams:
                bigrams.append(bigram)
            last_word = trigram[-1]

            if not last_word in vocabulary:
                vocabulary.append(last_word)

            if (bigram,last_word) not in count_matrix_dict:
                count_matrix_dict[bigram,last_word] = 0

            count_matrix_dict[bigram,last_word] += 1

    count_matrix = np.zeros((len(bigrams), len(vocabulary)))
    for trigram_key, trigram_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]), vocabulary.index(trigram_key[1])] = trigram_count

    return count_matrix , bigrams , vocabulary


#This function will predict the output given the input sequence.
def predict_next(count_matrix , bigrams , vocabulary , inp_str):
    if inp_str not in bigrams:
        return "sent not in vocabulary"
    idx = bigrams.index(inp_str)
    v_id = np.argmax(count_matrix[idx,:])
    word = vocabulary[v_id]
    return word
