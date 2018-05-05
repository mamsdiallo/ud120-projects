#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from plot_poi import plot_heatmap
from select_features import Select_k_best
from plot_poi import plot_kbest
from explore import get_incompletes, display_examples,build_df,display_NaN

### Task 1: Select what features you'll use.
# List of features
'''
1. bonus
2. deferral_payments
3. deferred_income
4. director_fees
5. email_address
6. exercised_stock_options
7. expenses
8. from_messages
9. from_poi_to_this_person
10. from_this_person_to_poi
11. loan_advances
12. long_term_incentive
13. other
14. poi
15. restricted_stock
16. restricted_stock_deferred
17. salary
18. shared_receipt_with_poi
19. to_messages
20. total_payments
21. total_stock_value
'''
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi" since it is the target.
features_list = ['poi',
                'bonus',
                'deferral_payments',
                'deferred_income',
                'director_fees',
                'exercised_stock_options',
                'expenses',
                'from_messages',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'loan_advances',
                'long_term_incentive',
                'other',
                'restricted_stock',
                'restricted_stock_deferred',
                'salary',
                'shared_receipt_with_poi',
                'to_messages',
                'total_payments',
                'total_stock_value'] 

# counting the number of features ?
print "number of features", len(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "total number of data points", len(data_dict)


### Task 2: Remove outliers

# Get the list of persons with more than 90% incomplete information
print "\n incompletes more than 90% incomplete information:\n",get_incompletes(data_dict,0.9)

display_examples(data_dict)

data = featureFormat(data_dict, features_list)
import re 
# patterns: "LAY", "TOTAL", "SKILLING"
LAY = re.compile(r'Lay',re.IGNORECASE)
TOTAL = re.compile(r'TOTAL',re.IGNORECASE)
SKILLING = re.compile(r'SKILLING',re.IGNORECASE)
AGENCY = re.compile(r'AGENCY',re.IGNORECASE)
WESLEY = re.compile(r'Wesley',re.IGNORECASE)
BHATNAGAR = re.compile(r'BHATNAGAR',re.IGNORECASE)

def display_info(dataset,pattern,feature):
    """Return the data related to the matching name if any."""
    for name in dataset.keys():
        if re.search(pattern,name):
            print "Match:",name
            print "[",feature,"] = ",dataset[name][feature]

# print some data points: LAY for total_payments
var = "total_payments"
display_info(data_dict,LAY,var)
# print exercised_stock_options for LAY
var = "exercised_stock_options"
display_info(data_dict,LAY,var)
# print SALARY for: TOTAL, SKILLING, AGENCY
var = "salary"
display_info(data_dict,TOTAL,var)
display_info(data_dict,SKILLING,var)
display_info(data_dict,AGENCY,var)
# print bonus for: TOTAL
var = "bonus"
display_info(data_dict,TOTAL,var)
# print from_messages for WESLEY
var = "from_messages"
display_info(data_dict,WESLEY,var)
# print poi for BHATNAGAR
var = "poi"
display_info(data_dict,BHATNAGAR,var)
var = "restricted_stock"
display_info(data_dict,BHATNAGAR,var)

# Looking for negative restricted_stock: might be an outlier
print "\n negative restricted_stock:\n"
for kk in data_dict:
    if (float(data_dict[kk]['restricted_stock']) < 0):
        print(kk)

# Looking for high total payment
print "\nhigh total payment:\n"
for kk in data_dict:
    if (float(data_dict[kk]['total_payments']) > 100000000):
        print(kk)

# Looking for high exercised_stock_options
print "\nhigh exercised_stock_options:\n"
for kk in data_dict:
    if (float(data_dict[kk]['exercised_stock_options']) > 30000000):
        print(kk)

# Let's catch the highest salary and highest bonus
print "\nhigh salary:\n"
for kk in data_dict:
    if (float(data_dict[kk]['salary']) > 20000000):
        print(kk)

print "\nhigh bonus:\n"
for kk in data_dict:
    if (float(data_dict[kk]['bonus']) > 80000000):
        print(kk)

# removing the outlier called TOTAL
data_dict.pop( "TOTAL", 0 )
# not a person and many missing information above 90%
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 )
# many missing information above 90%

import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set_style('whitegrid')
plt.style.use('classic')


# create a dataframe from dict
import pandas as pd

# build function data frame:
df = build_df(data_dict)

# display columns
df.columns

# After removal, we have the following graph:
sns.jointplot(x="bonus", y="salary", data=df)
plt.show()

print "total number of data points", len(data_dict)

### Checking the target class
count_poi = pd.value_counts(df['poi'], sort = True).sort_index()
print "number of poi:\n",count_poi

## Percentage of POI:
print "Percentage of POI ",100*18./144,"%"
df.hist(figsize=(20,15),bins=50)

### Task 3: Create new feature(s)

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

add_fraction_to_dict(data_dict, "perc_from_poi", "from_poi_to_this_person", "to_messages")
add_fraction_to_dict(data_dict, "perc_to_poi", "from_this_person_to_poi", "from_messages")

perc_from_poi  = [data_dict[key]['perc_from_poi'] for key in data_dict.keys()]
perc_to_poi  = [data_dict[key]['perc_to_poi'] for key in data_dict.keys()]

df['perc_from_poi'] = perc_from_poi
df['perc_to_poi'] = perc_to_poi
  
sns.jointplot(x="perc_from_poi", y="perc_to_poi", data=df)
plt.savefig('perc_from_poiVSperc_to_poi.png')
plt.show()

# Missing values
print "\ninfo:\n",df.info()

# Percentage of NaN values in data frame
display_NaN(df)

# Missing values : do not include these features
df.drop("loan_advances", axis=1,inplace=True)
df.drop("restricted_stock_deferred",axis=1,inplace=True)
df.drop("director_fees",axis=1,inplace=True)
df.drop("deferral_payments",axis=1,inplace=True) 

df.columns.values

# heatmap
plot_heatmap(df)

print "\ndata description:\n",df.describe()

# to make this notebook's output identical at every run
import numpy as np
np.random.seed(42)

# 17 features in total
features_list = ['poi',
                 'bonus',
                 'deferred_income',
                 'exercised_stock_options',
                 'expenses',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'salary',
                 'total_payments',
                 'total_stock_value',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'to_messages',
                 'perc_from_poi',
                 'perc_to_poi']


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

print("Number train dataset: ", len(features_train))
print("Number test dataset: ", len(features_test))
print("Total number: ", len(features_train)+len(features_test))

corr_matrix = df.corr()
corr_matrix["poi"].sort_values(ascending=False)


# Prepare the data for Machine Learning algorithms
# Select features
### Selection of features and use of SelectKBest
#from sklearn.feature_selection import SelectKBest

myList = Select_k_best(data_dict,features_list,13)
print "list of features",myList

# save the names and their respective scores separately
# reverse the tuples to go from most frequent to least frequent 
plot_kbest(myList)

# select best features from kbest
features_list = ['poi',
                'exercised_stock_options',
                'total_stock_value',
                'bonus',
                'salary',
                'perc_to_poi',
                'deferred_income',
                'long_term_incentive',
                'restricted_stock',
                'total_payments',
                'shared_receipt_with_poi',
                'expenses',
                'from_poi_to_this_person',
                'other',
                'perc_from_poi',
                'from_this_person_to_poi',
                'to_messages',
                'from_messages']

# select best features and getting rid of correlated features: 13 are remaining in total
features_list = ['poi',
                'exercised_stock_options',
                'bonus',
                'salary',
                'perc_to_poi',
                'deferred_income',
                'long_term_incentive',
                'total_payments',
                'shared_receipt_with_poi',
                'expenses',
                'from_poi_to_this_person',
                'perc_from_poi',
                'from_this_person_to_poi',
                'from_messages']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### split data into training and testing datasets
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

print("Number train dataset: ", len(features_train))
print("Number test dataset: ", len(features_test))
print("Total number: ", len(features_train)+len(features_test))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from tester import test_classifier

### trial with Naive Bayes for prediction
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(features_train, labels_train)

test_classifier(nb_clf, my_dataset, features_list)

### trial with Decision Tree for prediction
from sklearn.tree import DecisionTreeClassifier

DTree_clf = DecisionTreeClassifier(random_state=42)
DTree_clf.fit(features_train,labels_train)

test_classifier(DTree_clf, my_dataset, features_list)
print "Feature importances:",DTree_clf.feature_importances_
print "The number of classes ",DTree_clf.n_classes_
from sklearn import tree 
dot_data = tree.export_graphviz(DTree_clf,
                                feature_names=features_list[1:],
                                class_names=True,
                                label='all',
                                filled=True, rounded=True,
                                out_file="tree.dot")

# Use commands for generating image DecisionTree01.png
# >activate DAND
# >cd "~/final_project" # where the tree.dot is located
# >dot -Tpng tree.dot -o DecisionTree01.png

### trial with Random Forest for prediction
from sklearn.ensemble import RandomForestClassifier

RF_clf = RandomForestClassifier(criterion='entropy',max_features=1,
                             random_state=42)
RF_clf.fit(features_train, labels_train)

test_classifier(RF_clf, my_dataset, features_list)

print "Feature importances:",RF_clf.feature_importances_

### trial with AdaBoost model for prediction
from sklearn.ensemble import AdaBoostClassifier
depth = 10
aboost_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         algorithm="SAMME")    
aboost_clf.fit(features_train,labels_train)

test_classifier(aboost_clf, my_dataset, features_list)
print "Feature importances:",aboost_clf.feature_importances_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Tuning NaiveBayes Model
# There is no parameters for Naive Bayes except playing with the number of features. 
# scores to beat: Accuracy: 0.83993	Precision: 0.37073	Recall: 0.28750	F1: 0.32385	F2: 0.30102

# Tuning: 
# select One feature only: better performance when reducing the number of features
features_list = ['poi','exercised_stock_options']

### Naive Bayes for prediction
nb_clf = GaussianNB()
nb_clf.fit(features_train, labels_train)

## Evaluation on the final model
test_classifier(nb_clf, my_dataset, features_list)


### Tuning DecisionTreeClassifier
# Feature importances: [ 0.06620183,0.,0.04767267,0.17102017,0.05720721,0.02026089,0.06674174,
#0.2240468,0.21336521,0.08581081,0.,0.04767267,0.] from previous run of Decision Tree
# scores to beat: Accuracy: 0.81713	Precision: 0.30701	Recall: 0.29550	F1: 0.30115	F2: 0.29773

# Tuning:
# select the 9 most important features from last run of DecisionTreeClassifier
# Use of feature importances
features_list = ['poi',
                'exercised_stock_options',
                'salary',
                'perc_to_poi',
                'deferred_income',
                'total_payments',
                'shared_receipt_with_poi',
                'expenses',
                'from_poi_to_this_person',
                'from_this_person_to_poi']

from sklearn.model_selection import GridSearchCV

# Tuning with the following hyperparameters: 
# criterion, max_depth, min_samples_split, max_features 
param_grid = {'criterion': ["entropy"],
              'max_depth':[2,5,10],
              'min_samples_split':[2,3,4,5],
              'max_features': [1,2,3,4,5,6,7,8,9],
              'random_state':[42]}
DTree_clf = DecisionTreeClassifier()
#  F1 as scoring since the recall and the precision are equally important.
grid_search = GridSearchCV(DTree_clf, param_grid, cv=10, verbose=1, n_jobs=1,scoring='f1')
grid_search.fit(features_train, labels_train)

# Best hyperparameters
grid_search.best_params_

# Best Estimator
DTree_clf = grid_search.best_estimator_

grid_search.best_score_

# reuse of best estimator
DTree_clf.fit(features_train, labels_train)

## Evaluation on the final model
test_classifier(DTree_clf, my_dataset, features_list)

print "Feature importances:",DTree_clf.feature_importances_
print "The number of classes ",DTree_clf.n_classes_

dot_data = tree.export_graphviz(DTree_clf,
                                feature_names=features_list[1:],
                                class_names=True,
                                label='all',
                                filled=True, rounded=True,
                                out_file="tree2.dot")

# Use commands to generate image file DecisionTree02.png 
# >activate DAND
# >cd "~/final_project" # where the tree2.dot is located
# >dot -Tpng tree2.dot -o DecisionTree02.png

## Tuning RandomForestClassifier

# Accuracy to beat: Accuracy: 0.86247	Precision: 0.43907	Recall: 0.11350	F1: 0.18037	F2: 0.13326

'''
Feature importances from the previous run of Random Forest: 
[ 0.16600026  0.06917332  0.07329676  0.14227899  0.0484351   0.10840532
  0.09809793  0.04676382  0.08431206  0.03344624  0.05901818  0.02555952
  0.04521251]
'''
# Tuning: 
# Select the most important features from last run of Random Forest
# reducing the number of features from 9 to 2 since the performance was not sifficient
# selecting feature 'bonus' instead of 'perc_to_poi' for better performance 
features_list = ['poi',
                'exercised_stock_options',
                'bonus']

# Tuning with the following hyperparameters: 
# criterion, n_estimators, max_depth, max_features 
param_grid = {'criterion': ["gini", "entropy"],
              'n_estimators':[2,3,4,5],
              'max_depth':[None,5,10,15],
              'max_features': [None,1,2],
              'random_state':[42]}
RF_clf = RandomForestClassifier()
#  F1 as scoring since the recall and the precision are equally important.
grid_search = GridSearchCV(RF_clf, param_grid, cv=10, verbose=1, n_jobs=1,scoring='f1')
grid_search.fit(features_train, labels_train)


# Best hyperparameters
grid_search.best_params_

# Best estimator
RF_clf = grid_search.best_estimator_

grid_search.best_score_

# Reuse of the best estimator
RF_clf.fit(features_train, labels_train)

## Evaluation on the final model
test_classifier(RF_clf, my_dataset, features_list)

### Tuning Adaboost Model
# The scores to beat from last run of Adaboost model: Accuracy: 0.81860	Precision: 0.31195	Recall: 0.29900	F1: 0.30534	F2: 0.30150 

'''
Feature importances from previous run of Adaboost: 

[ 0.06620183 , 0., 0., 0.17102017, 0., 0.1060717, 0.16208709, 0.2240468, 0.21336521, 0., 0., 0.05720721, 0.]
'''
# Tuning:
# select best features and using feature importance from last run of Adaboost 
features_list = ['poi',
                'exercised_stock_options',
                'perc_to_poi',
                'long_term_incentive',
                'total_payments',
                'shared_receipt_with_poi',
                'expenses',
                'from_this_person_to_poi']

# Tuning with the following hyperparameters: 
# n_estimators, algorithm, learning_rage 
param_grid = {'n_estimators':[2,3,4,5,10],
              'algorithm':['SAMME'],
              'learning_rate':[0.5,1,2],
              'random_state':[42]}
Aboost_clf = AdaBoostClassifier()
#  F1 as scoring since the recall and the precision are equally important.
grid_search = GridSearchCV(Aboost_clf, param_grid, cv=10, verbose=1, n_jobs=1,scoring='f1')
grid_search.fit(features_train, labels_train)

# Best hyperparameters 
grid_search.best_params_

# Best estimator
clf = grid_search.best_estimator_

# reuse of the best estimator
clf.fit(features_train,labels_train)

## Evaluation on the final model
test_classifier(clf, my_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)