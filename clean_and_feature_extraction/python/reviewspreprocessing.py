import numpy as np
import pandas as pd
import textblob
import matplotlib.pyplot as plt

class ReviewsPreprocessing:
    
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
    def extract_feature(self, r, merged=False):

        if merged == False:
            r['comments'] = r['comments'].astype(str)
            r = r.groupby(by='listing_id')['comments'].sum().reset_index()
        else:
            pass

        r['sentiments'] = r.comments.apply(lambda x: textblob.TextBlob(x).sentiment)

        r[['polarity', 'subjectivity']] = r.sentiments.apply(pd.Series)
        r = r.drop(columns=['comments', 'sentiments'])
        
        return r
    
    # Do preprocessing
    def do_preprocessing(self, operation='extraction'):
        r = self.load()
        r = self.clean(r)
        if operation == 'extraction':
            r = self.extract_feature(r)
        elif operation == 'clean':
            pass
        else:
            raise ValueError('Operation can only be clean or extraction!')
        return r