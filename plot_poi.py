# -*- coding: utf-8 -*-
"""
Created on Sat May 05 12:44:53 2018

@author: Diallo
"""
# heatmap
# Plot the Pearson's Correlation Diagram: 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Settings
sns.set_style('whitegrid')
plt.style.use('classic')

def plot_heatmap(df):
    df2 = df[['bonus',
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
                    'perc_to_poi']]
    
    colormap = plt.cm.viridis
    plt.figure(figsize=(12,12))
    plt.title("Pearson's Correlation of Features", y=1.05, size=15)
    sns.heatmap(df2.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('PearsonCorrelationOfFeatures.png', dpi=100)    
    #plt.savefig('PearsonCorrelationOfFeatures.png')
    plt.show()
    
def plot_kbest(myList):
    # save the names and their respective scores separately
    # reverse the tuples to go from most frequent to least frequent 
    feat = zip(*myList)[0]
    score = zip(*myList)[1]
    y_pos = np.arange(len(feat)) 
    
    # calculate slope and intercept for the linear trend line
    #slope, intercept = np.polyfit(x_pos, score, 1)
    plt.barh(y_pos,score, align='center')
    #plt.yticks(y_pos, feat,rotation=90) 
    plt.yticks(y_pos, feat) 
    plt.xlabel('Importance Score')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('FeatureImportance.png', dpi=100)    
    
#    plt.savefig('FeatureImportance.png')
    plt.show()
              
        
def plotBonusvsSalary(df):
    sns.jointplot(x="bonus", y="salary", data=df)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('bonusVSsalary.png', dpi=100)    
    #plt.savefig('bonusVSsalary.png')
    plt.show()
   
def plotPerc_from_poivsPerc_to_poi(df):    
    sns.jointplot(x="perc_from_poi", y="perc_to_poi", data=df)
    plt.savefig('perc_from_poiVSperc_to_poi.png')
    plt.show()
    