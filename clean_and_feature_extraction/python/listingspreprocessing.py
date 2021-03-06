import pandas as pd
import numpy as np
import re, collections
from scipy.stats import boxcox
from geopy import distance

class ListingsPreprocessing:
    
    # Init
    def __init__(self, path):
        self.path = path

    # Load data
    def load(self):
        return pd.read_csv(self.path, low_memory=False)

    # Get attributes from l
    def get_attributes(self, l):
        return [(i, l.columns[i]) for i in range(len(l.columns))]

    # Prune unusable attributes
    def prune_attributes(self, l):
        l = l.dropna(how = 'any', subset = ['id','host_since','neighbourhood_cleansed',
        'latitude','longitude','property_type','room_type','accommodates','bathrooms',
        'bedrooms','beds','bed_type','amenities','price','cleaning_fee',
        'guests_included', 'extra_people','availability_30', 'availability_60', 'availability_90',
        'availability_365', 'first_review', 'last_review','minimum_nights','maximum_nights',
        'number_of_reviews','cancellation_policy'])
        for x in ['beds','bedrooms','bathrooms','price','accommodates']:
            l = l[l[x]!=0]
        l = l.reset_index(drop=True)
        for x in ['price','cleaning_fee','extra_people']:
            l[x] = l[x].str.replace(r'[^-+\d.]', '').astype(float)
        for x in ['host_since','first_review', 'last_review']:
            l[x] = pd.to_datetime(l[x])
        l.loc[l['maximum_nights']>2000,'maximum_nights']=2000
        l['no_reviews'] = 0
        l.loc[l['review_scores_rating']!=l['review_scores_rating'],['review_scores_rating','no_reviews']]=0,1
        return l

    # Encode categorical attributes
    def encode_cat(self, l):
        #for i in range(len(l)):
        #    if l.loc[i, 'review_scores_accuracy'] != l.loc[i, 'review_scores_accuracy']:
        #        l.loc[i, 'review_scores_accuracy'] = 'No Review'

        categorical_attributes = ['neighbourhood_cleansed', 
                              'property_type',
                              'room_type',
                              'bed_type',
                              #'review_scores_accuracy',
                              'cancellation_policy']

        for attr in categorical_attributes:
            encoded_cols = pd.get_dummies(l[attr])
            l = pd.concat((l.drop(attr, axis = 1), encoded_cols), axis = 1)

        return l

    # Standardize float attributes
    def standardize(self, col):
        mean = np.mean(col)
        std = np.std(col)
        return col.apply(lambda x: (x - mean) / std)
    
    #another normalize function   
    def normalize(self,col):
        newcol,la = boxcox(col)
        min_ = newcol.min()
        dif_ = newcol.max()-min_
        return (newcol-min_)/dif_
    
    # Encode uncategorical attributes
    def encode_uncat(self, l):
        noncategorical_attributes = ['host_since', 'accommodates', 
                                 'bedrooms', 'beds', 'bathrooms', 'number_of_reviews','review_scores_rating', 'cleaning_fee',
                                   'guests_included', 'extra_people','availability_30', 'availability_60', 'availability_90',
                                   'availability_365', 'first_review', 'last_review',
                                'minimum_nights', 'maximum_nights','latitude','longitude','attractions_in_1mile']
        dtattr = ['host_since','first_review', 'last_review']
        #for i in dtattr:
        #    min_=l[i].min()
        #    l[i] = (l[i]-min_).dt.days
       
        for attr in noncategorical_attributes:
            #if attr == 'host_since':
                #l[attr] = self.standardize(l[attr].str.replace(r'-', '').astype(float))
            #else:
                #l[attr] = self.standardize(l[attr].astype(float))
            print(attr)
            if attr in dtattr:
                min_ = l[attr].min()
                l[attr] = (l[attr]-min_).dt.days
            elif attr in ['latitude','longitude']:
                min_ = l[attr].min()
                l[attr] = l[attr]-min_
            #print(attr,sorted(l[attr])[:10])
            l[attr] = self.normalize(l[attr]+1)
            #print(sorted(l[attr])[:5],sorted(l[attr])[-5:])
        return l

    # Extract features from amenities
    def count_amenities(self, l):
        numa = 20
        def reformat(col):
            return col.apply(lambda x: x.strip('{}').replace('"','').split(','))

        def sort_by_value(d): 
            items=d.items() 
            backitems=[[v[1],v[0]] for v in items] 
            backitems.sort(reverse=True) 
            return [ backitems[i][1] for i in range(0,len(backitems))] 

        l['amenities'] = reformat(l['amenities'])
        amenity_lists = []
        for j in range(len(l)):
            amenity_lists.extend(l.loc[j, 'amenities'])
        frequency = collections.defaultdict(int)

        for amenity in amenity_lists: 
            frequency[amenity] += 1
        amenities_picked = sort_by_value(frequency)[0:numa]
        new_cols = pd.DataFrame(columns = amenities_picked, data = list(np.zeros((len(l), numa))), dtype=int)

        for j in range(len(new_cols)):
            for i in range(numa):
                if amenities_picked[i] in l.amenities[j]:
                    new_cols.iloc[j,i] += 1
        l = pd.concat((l.drop('amenities', axis=1),new_cols), axis=1)

        return l

    def count_attr(self,l):
        def num_attr(pos,attr):
            dis = attr.apply(lambda x: distance.distance(x[1:],pos).miles,axis=1)
            return (dis<1).sum()
        attractions = pd.read_excel('../seattle_attractions.xlsx')
        l['attractions_in_1mile'] = l[['latitude','longitude']].apply(lambda x: num_attr(x,attractions),axis=1)
        return l

    # Clean data
    def clean(self, l, attr):
        l_info = ['id',
        'host_since',
        'neighbourhood_cleansed',
        'latitude',
        'longitude',
        'property_type',
        'room_type',
        'accommodates',
        'bathrooms',
        'bedrooms',
        'beds',
        'bed_type',
        'amenities',
        'price',
        'cleaning_fee',
        'guests_included', 
        'extra_people',
        'availability_30',
        'availability_60',
        'availability_90',
        'availability_365', 
        'first_review', 
        'last_review',
        'minimum_nights',
        'maximum_nights',
        'number_of_reviews',
        'review_scores_rating',
        #'review_scores_accuracy',
        'cancellation_policy']

        l = l[l_info].copy()
        l = self.prune_attributes(l)
        return l

    # Extract features
    def extract_feature(self, l):
        l = self.count_attr(l)
        l = self.encode_cat(l)
        l = self.encode_uncat(l)
        l = self.count_amenities(l)
        return l

    # Preprocessing listings
    def do_preprocessing(self, operation='extraction'):
        raw_l = self.load()
        attributes = self.get_attributes(raw_l)
        l = self.clean(raw_l, attributes)
        if operation == 'extraction':
            l = self.extract_feature(l)
        elif operation == 'clean':
            pass
        else:
            raise ValueError('Operation can only be clean or extraction!')
        return l
