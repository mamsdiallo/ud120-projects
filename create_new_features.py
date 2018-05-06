# -*- coding: utf-8 -*-
"""
Created on Fri May 04 21:29:07 2018

@author: Diallo
"""

# creating new features
def compute_fraction(numerator, denominator):
    if numerator == "NaN" or numerator == 0 or denominator == "NaN" or denominator == 0:
        return 0.
    else:
        return float(numerator)/denominator
    
def add_fraction_to_dict(data_dict, new_feature_name, numerator_feature, denominator_feature):
    """
    Adds a new feature(corresponding to the division between two existing features) to the data dictionary
    :param data_dict: ictionary where each key is a string with a person's name and the value is another
    dictionary with the features associated to that person.
    :param new_feature_name: string containing the new feature name
    """
    for name, features in data_dict.iteritems():
        numerator = features[numerator_feature]
        denominator = features[denominator_feature]
        fraction = compute_fraction(numerator, denominator)
        features[new_feature_name] = fraction
                
def create_new_features(data_dict,df):
    add_fraction_to_dict(data_dict, "perc_from_poi", "from_poi_to_this_person", "to_messages")
    add_fraction_to_dict(data_dict, "perc_to_poi", "from_this_person_to_poi", "from_messages")
    
    perc_from_poi  = [data_dict[key]['perc_from_poi'] for key in data_dict.keys()]
    perc_to_poi  = [data_dict[key]['perc_to_poi'] for key in data_dict.keys()]
    
    df['perc_from_poi'] = perc_from_poi
    df['perc_to_poi'] = perc_to_poi
      
def drop_features(df):
    # Missing values : do not include these features
    df.drop("loan_advances", axis=1,inplace=True)
    df.drop("restricted_stock_deferred",axis=1,inplace=True)
    df.drop("director_fees",axis=1,inplace=True)
    df.drop("deferral_payments",axis=1,inplace=True)      