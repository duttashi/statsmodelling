from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

'''
Data Engineering & Analysis
'''

gapdata= pd.read_csv("gap.csv", low_memory=False)
data_clean=gapdata.dropna()

data_clean['breastcancerper100th']= data_clean['breastcancerper100th'].convert_objects(convert_numeric=True)
data_clean['femaleemployrate']= data_clean['femaleemployrate'].convert_objects(convert_numeric=True)
data_clean['alcconsumption']= data_clean['alcconsumption'].convert_objects(convert_numeric=True)
#data_clean['incomeperperson']= data_clean['incomeperperson'].convert_objects(convert_numeric=True)

#Create binary Breast Cancer Rate
def bin2cancer (row):
   if row['breastcancerper100th'] <= 20 :
      return 0
   elif row['breastcancerper100th'] > 20 :
      return 1

#Create binary Alcohol consumption
def bin2alcohol(row):
   if row['alcconsumption'] <= 5 :
      return 0
   elif row['alcconsumption'] > 5 :
      return 1
# create binary Female employee rate
def bin2femalemployee(row):
   if row['femaleemployrate'] <= 50 :
      return 0
   elif row['femaleemployrate'] > 50 :
      return 1
#Apply the new variable bin2alcohol to the gapmind dataset
data_clean['bin2femalemployee'] = data_clean.apply (lambda row: bin2femalemployee (row),axis=1)
data_clean['bin2alcohol'] = data_clean.apply (lambda row: bin2alcohol (row),axis=1)
data_clean['bin2cancer']=data_clean.apply(lambda row: bin2cancer(row),axis=1)

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors=data_clean[['bin2alcohol','bin2femalemployee']]

#target=data_clean.breastcancerper100th
target=data_clean.bin2cancer
print data_clean.dtypes
print data_clean.describe()
pred_train,pred_test,tar_train,tar_test=train_test_split(predictors,target,test_size=0.4)

print "predictor train shape: ",pred_train.shape
print "predictor test shape: ",pred_test.shape
print "target train shape: ",tar_train.shape
print "target test shape: ",tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
#classifier=classifier.fit(pred_train[~np.isnan(pred_train)],tra_train[~np.isnan(tra_train)])
classifier=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
from io import BytesIO as StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())

with open('gapDT-1.png', 'wb') as f:
    f.write(graph.create_png())
