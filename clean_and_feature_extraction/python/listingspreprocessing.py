import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, collections

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
        l['price'] = (l['price'].str.replace(r'[^-+\d.]', '').astype(float))
        l = l.dropna(how = 'any', subset = ['id', 'property_type', 'neighbourhood_cleansed', 'bathrooms', \
                                                  'bathrooms', 'beds', 'price'])
        l = l[l['beds']!=0]
        l = l[l['bedrooms']!=0]
        l = l[l['bathrooms']!=0]
        l = l[l['price']!=0]
        l = l[l['accommodates']!=0]
        l = l.reset_index(drop=True)
        return l

    # Encode categorical attributes
    def encode_cat(self, l):
        for i in range(len(l)):
            if l.loc[i, 'review_scores_accuracy'] != l.loc[i, 'review_scores_accuracy']:
                l.loc[i, 'review_scores_accuracy'] = 'No Review'

        categorical_attributes = ['neighbourhood_cleansed', 
                              'property_type',
                              'room_type',
                              'bed_type',
                              'review_scores_accuracy',
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

    # Encode uncategorical attributes
    def encode_uncat(self, l):
        noncategorical_attributes = ['host_since', 'accommodates', 
                                 'bedrooms', 'beds', 'bathrooms', 'number_of_reviews',
                                'minimum_nights', 'maximum_nights']
#         for attr in noncategorical_attributes:
#             if attr == 'host_since':
#                 l[attr] = self.standardize(l[attr].str.replace(r'-', '').astype(float))
#             else:
#                 l[attr] = self.standardize(l[attr].astype(float))
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

    # Clean data
    def clean(self, l, attr):
        l_info = ['id',
#         'host_since',
        'neighbourhood_cleansed',
        'property_type',
        'room_type',
        'accommodates',
        'bathrooms',
        'bedrooms',
        'beds',
        'bed_type',
        'amenities',
        'price',
        'minimum_nights',
        'maximum_nights',
        'number_of_reviews',
        'review_scores_accuracy',
        'cancellation_policy']

        l = l[l_info].copy()
        l = self.prune_attributes(l)
        return l

    # Extract features
    def extract_feature(self, l):
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