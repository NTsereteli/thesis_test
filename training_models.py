import pandas as pd
from utils.data import Data
from itertools import product

df = pd.read_csv('political_posts.csv')
data = Data(df)
dfs = data.time_series_split()
tagged_doc = list(data.tagged_document(dfs[0]['content'].apply(data.preprocess_doc)))


params = {'vector_size':[8, 16, 32, 64, 128, 256, 512, 1024],
          'max_vocab_size':[10**3, 10**4, 10**5, 10**6],
          'epochs':[10, 20, 50]}
param_combinations = list(product(*params.values()))

for current_params in param_combinations:
    params.update(zip(params.keys(), current_params))
    data.train(tagged_doc, params)