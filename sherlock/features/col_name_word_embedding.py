import numpy  as np
import pandas as pd
import os
from scipy       import stats
from string import punctuation
from collections import OrderedDict
SHERLOCKPATH = os.environ['SHERLOCKPATH']

embedding_loc = os.path.join(SHERLOCKPATH, 'pretrained', 'glove.6B')

word_vectors_f = open(os.path.join(embedding_loc ,'glove.6B.50d.txt'))
word_to_embedding = {}
for l in word_vectors_f:
    term, vector = l.strip().split(' ', 1)
    vector = np.array(vector.split(' '), dtype=float)
    word_to_embedding[term] = vector


def split_string_by_uppercase_and_special_characters(string):
    substrings = []
    current_substring = ''

    for char in string:
        if not char.isupper() and not char.islower() and not char in punctuation + ' ':
            continue
        elif char.isupper() and current_substring == '':
            current_substring += char
        elif char.islower() and current_substring == '':
            current_substring += char
        elif char in punctuation + ' ' and current_substring == '':
            continue
        elif char.isupper() and current_substring.isupper():
            current_substring += char
        elif current_substring[0].isupper() and len(current_substring)==1 and char.islower():
            current_substring += char
        elif char.islower() and current_substring.isupper():
            substrings.append(current_substring)
            current_substring = char
        elif char.islower() and current_substring.islower():
            current_substring += char
        elif char.isupper() and current_substring.islower():
            substrings.append(current_substring)
            current_substring = char
        elif char.islower():
            current_substring += char
        elif char.isupper():
            substrings.append(current_substring)
            current_substring = char
        elif char in punctuation + ' ':
            substrings.append(current_substring)
            current_substring = ''
        else:
            current_substring += char

    if current_substring:
        substrings.append(current_substring)
    
    substrings=[s.lower() for s in substrings]

    return substrings


num_embeddings = 50
# Input: a col_name as single string
def extract_col_name_word_embeddings_features(col_name):
    
    f = OrderedDict()
    embeddings = []
    
    words = split_string_by_uppercase_and_special_characters(col_name)

    for w in words:
        if w in word_to_embedding:
            embeddings.append(word_to_embedding.get(w))
    if embeddings:
        mean_of_word_embeddings = np.nanmean(embeddings, axis=0)

    if len(embeddings) == 0: 
        for i in range(num_embeddings): f['col_name_word_embedding_{}'.format(i)]  = np.nan
        return f
    
    else:
        for i, e in enumerate(mean_of_word_embeddings):
            f['col_name_word_embedding_{}'.format(i)] = e
        return f