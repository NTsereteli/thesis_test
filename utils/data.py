import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from datetime import date

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
gensim.models.doc2vec.logger.setLevel(logging.INFO)

class Data:
    def __init__(self, df):
        self.df = df

    def time_series_split(self, margin1=None, margin2=None):
        self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        
        if margin1 is None:
            margin1 = date(year=2023, month=1, day=1)
        if margin2 is None:
            margin2 = date(year=2023, month=2, day=15)
        
        df1 = self.df[(self.df['date'] < margin1)]
        df2 = self.df[(self.df['date'] > margin1) & (self.df['date'] < margin2)]
        df3 = self.df[(self.df['date'] > margin2)]
        return df1, df2, df3
    
    def preprocess_doc(self, text):
        text = re.sub(r'[^ა-ჰ ]', ' ', text)
        text = re.sub(r' +', ' ', text)
        lst = text.split()
        return lst
    
    def tagged_document(self, list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield TaggedDocument(list_of_words, [i])

    def train(self, tagged_doc, params):
        model = Doc2Vec(**params)
        model.build_vocab(tagged_doc)
        model.train(tagged_doc, total_examples=model.corpus_count, epochs=model.epochs)
        filename = '__'.join([f'{key}_{val}' for key,val in params.items()])
        model.save(f"models/{filename}.model")
        return model
    
    def docs2vecs(self, model, list_of_list_of_words):
        vecs_list = []
        for list_of_words in list_of_list_of_words:
            vecs_list.append(model.infer_vector(list_of_words))
        vecs = np.vstack(vecs_list)
        return vecs

    def evaluate(self, model, dfs):
        logit = LogisticRegression(max_iter=100000)
        df1, df2, df3 = dfs
        y1, y2, y3 = df1['source'], df2['source'], df3['source']
        list_of_list_of_words2 = df2['content'].apply(self.preprocess_doc).tolist()
        list_of_list_of_words3 = df3['content'].apply(self.preprocess_doc).tolist()
        vecs2 = self.docs2vecs(model, list_of_list_of_words2)
        vecs3 = self.docs2vecs(model, list_of_list_of_words3)

        logit.fit(vecs2, y2)
        y_pred3 = logit.predict(vecs3)
        return f1_score(y3, y_pred3, average='micro')




        