import pandas as pd
from collections import OrderedDict
import numpy as np
import os
import random
from os.path import join
from extract.helpers import utils
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from topic_model import train_LDA
from utils import name2dic

TYPENAME = os.environ['TYPENAME']
# objects too large to pass for multiprocessing
LDA_name = os.environ['LDA_name']
model_loc = join(os.environ['BASEPATH'], 'topic_model', "LDA_cache", TYPENAME)
LDA = LdaModel.load(join(model_loc,'model_{}'.format(LDA_name)))
Dic = Dictionary.load(join(model_loc,'dictionary_{}'.format(LDA_name)))

n_samples = 1000
seed=0

def get_table_topic(df, lda, common_dict, model_name):
    # get topic vector for table
    kwargs = name2dic(model_name)

    table_seq = []
    for col in df.columns:
        processed_col = train_LDA.process_col(df[col], **kwargs)
        table_seq.extend(processed_col)
    
    vector = lda[common_dict.doc2bow(table_seq)]
    return [v[1] for v in vector]



def extract_topic_features(df_dic, sample=True):

    df, locator, dataset_id = df_dic['df'], df_dic['locator'], df_dic['dataset_id']
    table_features = OrderedDict()

    if sample:
        # sample if more than 1000 values in column
        n_values=len(df)
        if n_values > n_samples:
            n_values = n_samples
        df=df\
            .sample(n_values, random_state=seed) \
            .reset_index(drop=True) \
            .astype(str)

    try:
        vec_LDA = get_table_topic(df, LDA, Dic, LDA_name)

    except Exception as e:
        print('Table topic exception:', e)
        return

    table_features['locator'] = locator
    table_features['dataset_id'] = dataset_id
    table_features['table_topic'] = vec_LDA    

    return pd.DataFrame([table_features])
