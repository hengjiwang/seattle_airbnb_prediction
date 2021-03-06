import pandas as pd
import os,sys
from listingspreprocessing import ListingsPreprocessing
from reviewspreprocessing import ReviewsPreprocessing
from datepreprocessing import DatePreprocessing

class CreateDataset:
    def __init__(self, path):
        self.path = path

    def merge_listings(self, folders):
    
        print('merging listings...')
        l = ListingsPreprocessing(self.path+folders[0]+'/listings.csv').do_preprocessing('clean')
        
        for f in folders[1:]:
            l_new = ListingsPreprocessing(self.path+f+'/listings.csv').do_preprocessing('clean')
            l = pd.concat([l, l_new],ignore_index=True)
        
        l.drop_duplicates(subset='id', keep='first', inplace=True)
        l.sort_values(by = 'id', inplace = True)
            
        return l

    def merge_reviews(self, folders):
        
        print('merging reviews...')
        
        r = ReviewsPreprocessing(self.path+folders[0]+'/reviews.csv').do_preprocessing('clean')
        
        for f in folders[1:]:
            r_new = ReviewsPreprocessing(self.path+f+'/reviews.csv').do_preprocessing('clean')
            r = pd.concat([r, r_new],ignore_index=True)
        
        r.drop_duplicates(keep='first', inplace=True)
        r['comments'] = r['comments'].astype(str)
        r = r.groupby(by='listing_id')['comments'].sum().reset_index()
        r.sort_values(by = 'listing_id', inplace = True)
            
        return r

    def merge_calendars(self, folders):
        
        print('merging calendars...')
        
        df = DatePreprocessing(self.path+folders[0]+'/calendar.csv').clean_calendar()
        for f in folders[1:]:
            df2 = DatePreprocessing(self.path+f+'/calendar.csv').clean_calendar()
            df = pd.concat([df,df2],ignore_index=True)
        df = df.drop_duplicates(subset=['listing_id','date'], keep='first').reset_index(drop='True')
        return df

    def create_dataset(self,file_out):
        print('create dataset...')
        folders = [x for x in os.listdir(self.path)]
        l_merged = self.merge_listings(folders).reset_index(drop = True)
        r_merged = self.merge_reviews(folders).reset_index(drop = True)
        d_merged = self.merge_calendars(folders)
        print('extract listing features...')
        l_extracted = ListingsPreprocessing('').extract_feature(l_merged)
        print('extract review features...')
        r_extracted = ReviewsPreprocessing('').extract_feature(r_merged, merged=True)
        print('extrac date features...')
        d_extracted = DatePreprocessing('').extract_feature(d_merged)
        print('starting to merge...')
        data = pd.merge(l_extracted, r_extracted,
            left_on='id',right_on='listing_id',
            how='inner').drop(columns = 'listing_id')

        data = pd.merge(data.drop(columns='price'),
            d_extracted,left_on='id',right_on='listing_id',
            how='inner').drop(columns = 'listing_id')
        #print('normalize scraped...')
        
        data['days_from_scraped'] = ListingsPreprocessing('na').normalize(data['days_from_scraped']+1)
        data['aveT'] = ListingsPreprocessing('na').normalize(data['aveT'])
        data['precipitation'] = ListingsPreprocessing('na').normalize(data['precipitation'])      
        print('save to files...')
        data.to_csv('../../save/'+file_out, index=False)
        print('^O^ Finish, save to %s!' % file_out)

if __name__ == '__main__':
    fl = sys.argv[1]
    cd = CreateDataset('../../data/')
    cd.create_dataset(fl)

