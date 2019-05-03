import numpy as np
import pandas as pd
import textblob
import matplotlib.pyplot as plt

class reviewsPreprocessing:
    
    # Init
    def __init__(self, path):
        self.path = path
        
    # Load data
    def load(self):
        return pd.read_csv(self.path)

    # Clean data
    def clean(self, r):
        return r.drop(['id', 'date', 'reviewer_id', 'reviewer_name'], axis=1)
    
    # Extract sentimental features
    def extract_feature(self, r):
        ids = list(np.unique(r.listing_id.tolist()))

        for i in ids:
            new_row = pd.DataFrame(columns=['listing_id', 'comments'], data=[[i, ''.join(str(r.loc[r.listing_id == i].comments.tolist()))]])    
            r = r[r.listing_id != i]
            r = pd.concat((r, new_row))
            r = r.reset_index(drop=True)

        r['sentiments'] = r.comments.apply(lambda x: textblob.TextBlob(x).sentiment)

        r[['polarity', 'subjectivity']] = r.sentiments.apply(pd.Series)
        r = r.drop(columns=['comments', 'sentiments'])
        return r
    
    # Do preprocessing
    def do_preprocessing(self):
        r = self.load()
        r = self.clean(r)
        r = self.extract_feature(r)
        return r