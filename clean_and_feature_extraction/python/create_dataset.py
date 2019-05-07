import pandas as pd
from listingspreprocessing import ListingsPreprocessing
from reviewspreprocessing import ReviewsPreprocessing
from datepreprocessing import DatePreprocessing
import os, sys

class CreateDataSet:
    
    # Init
    def __init__(self, path, path_to_save):
        self.path1 = path + 'listings.csv'
        self.path2 = path + 'reviews.csv'
        self.path3 = path + 'calendar.csv'
        self.path_to_save = path_to_save

    
    def load_and_combine(self):
        l = ListingsPreprocessing(self.path1).do_preprocessing()
        r = ReviewsPreprocessing(self.path2).do_preprocessing()
        d = DatePreprocessing(self.path3).do_preprocessing()
        data = pd.merge(l, r,
            left_on='id',right_on='listing_id',
            how='inner').drop(columns = 'listing_id')
        data = pd.merge(data.drop(columns='price'),
            d,left_on='id',right_on='listing_id',
            how='inner').drop(columns = 'listing_id')
        return data
    
    def create(self):
        data = self.load_and_combine()
        data.to_csv(self.path_to_save)

if __name__ == '__main__':
    sub_lists = os.listdir(sys.argv[1])
    for sub in sub_lists:
        creater = CreateDataSet(sys.argv[1]+sub+'/', sys.argv[2]+'/'+sub+'.csv')
        print(sub)
        creater.create()
