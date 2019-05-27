import pandas as pd
import os
from listingspreprocessing import ListingsPreprocessing
from reviewspreprocessing import ReviewsPreprocessing
from datepreprocessing import DatePreprocessing

class CreateDataset:
    def __init__(self, path):
        self.path = path

    def merge_listings(self, folders):
    
        l = ListingsPreprocessing(self.path+folders[0]+'/listings.csv').do_preprocessing('clean')
        
        for f in folders[1:]:
            l_new = ListingsPreprocessing(self.path+f+'/listings.csv').do_preprocessing('clean')
            l = pd.concat([l, l_new],ignore_index=True)
        
        l.drop_duplicates(subset='id', keep='last', inplace=True)
        l.sort_values(by = 'id', inplace = True)
            
        return l

    def merge_reviews(self, folders):
    
        r = ReviewsPreprocessing(self.path+folders[0]+'/reviews.csv').do_preprocessing('clean')
        
        for f in folders[1:]:
            r_new = ReviewsPreprocessing(self.path+f+'/reviews.csv').do_preprocessing('clean')
            r = pd.concat([r, r_new],ignore_index=True)
        
        r.drop_duplicates(keep='last', inplace=True)
        r['comments'] = r['comments'].astype(str)
        r = r.groupby(by='listing_id')['comments'].sum().reset_index()
        r.sort_values(by = 'listing_id', inplace = True)
            
        return r

    def merge_calendars(self, folders):
    
        def clean_calendar(fl):
            df = pd.read_csv(fl).dropna().drop(columns='available').reset_index(drop=True)
            df = df[['listing_id','date','price']] #only keep these three columns
            df['price'] =  df['price'].apply(lambda x: x.replace('$','').replace(',','')).astype('float')   
            df['date'] = pd.to_datetime(df['date'])
            return df
        
        df = clean_calendar(self.path+folders[0]+'/calendar.csv')
        for f in folders[1:]:
            df = pd.concat([df,clean_calendar(self.path+f+'/calendar.csv')],ignore_index=True)
        df = df.groupby(by=['listing_id','date'])['price'].max().reset_index()
        return df

    def create_dataset(self):
        folders = [x for x in os.listdir(self.path)]
        l_merged = self.merge_listings(folders).reset_index(drop = True)
        r_merged = self.merge_reviews(folders).reset_index(drop = True)
        d_merged = self.merge_calendars(folders)
        l_extracted = ListingsPreprocessing('').extract_feature(l_merged)
        r_extracted = ReviewsPreprocessing('').extract_feature(r_merged, merged=True)
        d_extracted = DatePreprocessing('').extract_feature(d_merged)
        data = pd.merge(l_extracted, r_extracted,
            left_on='id',right_on='listing_id',
            how='inner').drop(columns = 'listing_id')

        data = pd.merge(data.drop(columns='price'),
            d_extracted,left_on='id',right_on='listing_id',
            how='inner').drop(columns = 'listing_id')
        data.to_csv('../../save/all_data.csv', index=False)


if __name__ == '__main__':
    cd = CreateDataset('../../data/')
    cd.create_dataset()
