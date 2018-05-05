# -*- coding: utf-8 -*-
"""
Created on Sat May 05 11:15:31 2018

@author: Diallo
"""

### Task 2: Remove outliers
import re 
import pandas as pd

# patterns: "TOTAL", "AGENCY"
TOTAL = re.compile(r'TOTAL',re.IGNORECASE)
AGENCY = re.compile(r'AGENCY',re.IGNORECASE)

def get_incompletes(dataset, threshold):
    """
    Returns an array of person names that have no information (NaN) in a
    percentage of features above a given threshold (between 0 and 1).
    """
    incompletes = []
    for person in dataset:
        n = 0
        for key, value in dataset[person].iteritems():
            if value == 'NaN' or value == 0:
                n += 1
        fraction = float(n) / float(21)
        if fraction > threshold:
            incompletes.append(person)
            print fraction*100.0," %"

    return incompletes

def display_info(dataset,pattern,feature):
    """Return the data related to the matching name if any."""
    for name in dataset.keys():
        if re.search(pattern,name):
            print "Name:",name
            print "[",feature,"] = ",dataset[name][feature]

def display_examples(data_dict):
    # print SALARY for: TOTAL, AGENCY
    # print poi for TOTAL
    var = "poi"
    display_info(data_dict,TOTAL,var)
    # print bonus for: TOTAL
    var = "bonus"
    display_info(data_dict,TOTAL,var)
    # print salary for: TOTAL
    var = "salary"
    display_info(data_dict,TOTAL,var)
    # print poi for AGENCY
    var = "poi"
    display_info(data_dict,AGENCY,var)
    # print bonus for AGENCY
    var = "bonus"
    display_info(data_dict,AGENCY,var)
    # print salary for AGENCY
    var = "salary"
    display_info(data_dict,AGENCY,var)

    # Let's catch the highest salary and highest bonus
    print "\nWho has the highest salary? "
    for kk in data_dict:
        if (float(data_dict[kk]['salary']) > 20000000):
            print(kk)

    print "\nWho has the highest bonus? "
    for kk in data_dict:
        if (float(data_dict[kk]['bonus']) > 80000000):
            print(kk)
            
# create a dataframe from dict
# build function data frame:
def build_df(data_dict):
    name = [key for key in data_dict.keys()]
    poi = [data_dict[key]['poi'] for key in data_dict.keys()]
    bonus = [data_dict[key]['bonus'] for key in data_dict.keys()]
    deferral_payments = [data_dict[key]['deferral_payments'] for key in data_dict.keys()]
    deferred_income = [data_dict[key]['deferred_income'] for key in data_dict.keys()]
    director_fees = [data_dict[key]['director_fees'] for key in data_dict.keys()]
    exercised_stock_options = [data_dict[key]['exercised_stock_options'] for key in data_dict.keys()]
    expenses = [data_dict[key]['expenses'] for key in data_dict.keys()]
    loan_advances = [data_dict[key]['loan_advances'] for key in data_dict.keys()]
    long_term_incentive = [data_dict[key]['long_term_incentive'] for key in data_dict.keys()]
    other = [data_dict[key]['other'] for key in data_dict.keys()]
    restricted_stock = [data_dict[key]['restricted_stock'] for key in data_dict.keys()]
    restricted_stock_deferred = [data_dict[key]['restricted_stock_deferred'] for key in data_dict.keys()]
    salary = [data_dict[key]['salary'] for key in data_dict.keys()]
    total_payments = [data_dict[key]['total_payments'] for key in data_dict.keys()]
    total_stock_value = [data_dict[key]['total_stock_value'] for key in data_dict.keys()]
    from_messages = [data_dict[key]['from_messages'] for key in data_dict.keys()]
    from_poi_to_this_person = [data_dict[key]['from_poi_to_this_person'] for key in data_dict.keys()]
    from_this_person_to_poi = [data_dict[key]['from_this_person_to_poi'] for key in data_dict.keys()]
    shared_receipt_with_poi = [data_dict[key]['shared_receipt_with_poi'] for key in data_dict.keys()]
    to_messages  = [data_dict[key]['to_messages'] for key in data_dict.keys()]
    
    values = [('name',name),
              ('poi',poi),
              ('bonus',bonus),
              ('deferral_payments',deferral_payments),
              ('deferred_income',deferred_income),          
              ('director_fees',director_fees),
              ('exercised_stock_options',exercised_stock_options),
              ('expenses',expenses),
              ('loan_advances',loan_advances),
              ('long_term_incentive',long_term_incentive),
              ('other',other),
              ('restricted_stock',restricted_stock),
              ('restricted_stock_deferred',restricted_stock_deferred),
              ('salary',salary),
              ('total_payments',total_payments),
              ('total_stock_value',total_stock_value),
              ('from_messages',from_messages),
              ('from_poi_to_this_person',from_poi_to_this_person),
              ('from_this_person_to_poi',from_this_person_to_poi),
              ('shared_receipt_with_poi',shared_receipt_with_poi),
              ('to_messages',to_messages)]

    df = pd.DataFrame.from_items(values)
    df.bonus = pd.to_numeric(df.bonus, errors='coerce')
    df.poi = pd.to_numeric(df.poi, errors='coerce')
    df.deferral_payments = pd.to_numeric(df.deferral_payments, errors='coerce')
    df.deferred_income = pd.to_numeric(df.deferred_income, errors='coerce')
    df.director_fees = pd.to_numeric(df.director_fees, errors='coerce')
    df.exercised_stock_options = pd.to_numeric(df.exercised_stock_options, errors='coerce')
    df.expenses = pd.to_numeric(df.expenses, errors='coerce')
    df.loan_advances = pd.to_numeric(df.loan_advances, errors='coerce')
    df.long_term_incentive = pd.to_numeric(df.long_term_incentive, errors='coerce')
    df.other = pd.to_numeric(df.other, errors='coerce')
    df.restricted_stock = pd.to_numeric(df.restricted_stock, errors='coerce')
    df.restricted_stock_deferred = pd.to_numeric(df.restricted_stock_deferred, errors='coerce')
    df.salary = pd.to_numeric(df.salary, errors='coerce')
    df.total_payments = pd.to_numeric(df.total_payments, errors='coerce')
    df.total_stock_value = pd.to_numeric(df.total_stock_value, errors='coerce')
    df.from_messages = pd.to_numeric(df.from_messages, errors='coerce')
    df.from_poi_to_this_person = pd.to_numeric(df.from_poi_to_this_person, errors='coerce')
    df.from_this_person_to_poi = pd.to_numeric(df.from_this_person_to_poi, errors='coerce')
    df.shared_receipt_with_poi = pd.to_numeric(df.shared_receipt_with_poi, errors='coerce')
    df.to_messages = pd.to_numeric(df.to_messages, errors='coerce')
    
    return df

# Percentage of NaN values in data frame
def PercentNaN(df,feature):
    f_NaN = df[feature].isnull().sum()*100.0/df.shape[0]
    print "% of NaN values for ",feature,"{0:.2f}".format(f_NaN)


# display NaN %    
def display_NaN(df):
    for title in df.columns:
        PercentNaN(df,title)
        