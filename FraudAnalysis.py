# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:23:43 2018

@author: Diallo
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# needed library
import pprint
import sys
import numpy as np
import pandas as pd

# settings
np.set_printoptions(threshold='nan')

# utilitaries 
sys.path.append("../tools/")
from feature_format import featureFormat

# size of the dataset
print "len:",len(enron_data)
 
# Conversion of dictionary into an array
f=['bonus','deferral_payments','deferred_income','director_fees',
   'exercised_stock_options','expenses',
   'from_messages','from_poi_to_this_person','from_this_person_to_poi',
   'loan_advances','long_term_incentive','other','restricted_stock',
   'restricted_stock_deferred','salary','shared_receipt_with_poi',
   'to_messages','total_payments','total_stock_value']


'''
EXPLORE 
'''
listF = featureFormat(enron_data,f,remove_NaN=False)
pprint.pprint(listF[:10])

# convert to data frame
df = pd.DataFrame(listF)

# First rows
pprint.pprint(df.head(n=10))

# last rows
print df.tail(n=10)

# Summary of data
print df.describe(include = 'all')

# the index values
print df.index

# Data frame summary 
print df.info()

# Save data frame into csv
df.to_csv("df.csv",index=False)

# MODEL K-MEAN
