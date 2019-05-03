import pandas as pd
from listingsPreprocessing import listingsPreprocessing
from reviewsPreprocessing import reviewsPreprocessing

class createDataSet:
    
    # Init
    def __init__(self, path_l, path_r):
        self.path1 = path_l
        self.path2 = path_r    
    
    def load(self):
        l = listingsPreprocessing(self.path1).do_preprocessing()
        r = reviewsPreprocessing(self.path2).do_preprocessing()
        return (l,r)
        
    def combine(self, l, r):
        for i in range(len(l)):
            idd = int(l.iloc[i].id)
            if idd in r.listing_id.tolist():
                l.loc[i, 'polarity'] = float(r[r.listing_id == idd].polarity)
                l.loc[i, 'subjectivity'] = float(r[r.listing_id == idd].subjectivity)
        return l
    
    def create(self):
        l,r = self.load()
        l = self.combine(l, r)
        return l