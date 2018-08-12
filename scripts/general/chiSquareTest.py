__author__ = 'Ashoo'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import warnings

sns.set(color_codes=True)
# Reading the data where low_memory=False increases the program efficiency
data= pd.read_csv("gapminder.csv", low_memory=False)
# setting variables that you will be working with to numeric
data['breastcancerper100th']= data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['femaleemployrate']= data['femaleemployrate'].convert_objects(convert_numeric=True)
data['alcconsumption']= data['alcconsumption'].convert_objects(convert_numeric=True)

#print "Showing missing data coulmn-wise"
#print data.isnull().sum()
# Create a copy of the original dataset as sub5 by using the copy() method
sub5=data.copy()
# Since the data is all continuous variables therefore the use the mean() for missing value imputation
sub5.fillna(sub5['breastcancerper100th'].mean(), inplace=True)
sub5.fillna(sub5['femaleemployrate'].mean(), inplace=True)
sub5.fillna(sub5['alcconsumption'].mean(), inplace=True)
# Showing the count of null values after imputation
#print sub5.isnull().sum()

# categorize quantitative variable based on customized splits using the cut function
sub5['alco']=pd.qcut(sub5.alcconsumption,6,labels=["0","1-4","5-9","10-14","15-19","20-24"])
sub5['brst']=pd.qcut(sub5.breastcancerper100th,5,labels=["1-20","21-40","41-60","61-80","81-90"])

# Converting response variable to categorical
sub5['brst']=sub5['brst'].astype('category')

# Cross tabulating the response variable with explantory variable
ct1=pd.crosstab(sub5['brst'],sub5['alco'])
#ct1=pd.crosstab(sub5['alco'],sub5['brst'])
print "Contigency Table"
print ct1
print "\n\n"
# the axis=0 statement tells python to sum all the values in each column in python
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# Chi-Square
print('\n\nChi-square value, p value, expected counts')
cs1=scipy.stats.chi2_contingency(ct1)
print(cs1)

sub5['brst']=sub5['brst'].astype('category')
sub5['alco']=sub5['alco'].convert_objects(convert_numeric=True)

#sns.factorplot(x='alcconsumption', y='breastcancerper100th', data=sub5, kind="bar", ci=None)
sns.factorplot(x='alco', y='brst', data=sub5, kind="bar",ci=None)
plt.xlabel("Alcohol consumption in Liters")
plt.ylabel("Breast Cancer cases per 100th women")
# ====================================================
# POST HOC COMPARISON TEST

recode2={1-20:1,21-40:2}
sub5['COMP1v2']=sub5['brst'].map(recode2)

ct2=pd.crosstab(sub5['brst'],sub5['COMP1v2'])

print "Contigency Table -2\n"
print ct2
print "\n\n"

# the axis=0 statement tells python to sum all the values in each column in python
colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)

# Chi-Square
print('\n\nChi-square value, p value, expected counts')
cs2=scipy.stats.chi2_contingency(ct2)
print(cs2)

#######################################################
recode3={41-60:3,61-80:4}
sub5['COMP1v3']=sub5['alco'].map(recode3)

ct3=pd.crosstab(sub5['brst'],sub5['COMP1v3'])
print "Contigency Table - 3\n"
print ct3
print "\n\n"
# the axis=0 statement tells python to sum all the values in each column in python
colsum=ct3.sum(axis=0)
colpct=ct3/colsum
print(colpct)

# Chi-Square
print('\n\nChi-square value, p value, expected counts')
cs3=scipy.stats.chi2_contingency(ct3)
print(cs3)



