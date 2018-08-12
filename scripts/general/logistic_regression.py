# Importing the required libraries
# Note %matplotlib inline works only for ipython notebook. It will not work for PyCharm. It is used to show the plot distributions
#%matplotlib inline
# To show the plot in PyCharm editor use the show()
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.multicomp as multi
import seaborn
import matplotlib.pyplot as plt
plt.interactive(False)
# Reading the data where low_memory=False increases the program efficiency
data= pd.read_csv("gapminder.csv", low_memory=False)

# setting variables that you will be working with to numeric
data['breastcancerper100th']=pd.to_numeric(data['breastcancerper100th'], errors='coerce')
data['alcconsumption']= pd.to_numeric(data['alcconsumption'], errors='coerce')
data['femaleemployrate']=pd.to_numeric(data['femaleemployrate'], errors='coerce')

# creating a copy of the dataframe
sub14=data.copy()
# Checking for missing values in the data frame
#print "\n\t Missing values count\n",sub14.isnull().sum()

# Since the data is all continuous variables therefore the use the mean() for missing value imputation
sub14.fillna(sub14['breastcancerper100th'].mean(), inplace=True)
sub14.fillna(sub14['femaleemployrate'].mean(), inplace=True)
sub14.fillna(sub14['alcconsumption'].mean(), inplace=True)
#print "\nMissing value imputation done\n",sub14.isnull().sum()

# Now creating a binary response variable for logistic regression
#Create binary Breast Cancer Rate
def bin2cancer (row):
   if row['breastcancerper100th'] <= 20 :
      return 0
   elif row['breastcancerper100th'] > 20 :
      return 1
#Apply the new variable bin2cancer to the gapmind dataset
data['bin2cancer']=data.apply(lambda row: bin2cancer(row),axis=1)

#Creat binary Income per person
def bin2income(row):
   if row['incomeperperson'] <= 5000 :
      return 0
   elif row['incomeperperson'] > 5000 :
      return 1
#Apply the new variable bin2income to the gapmind dataset
data['bin2income'] = data.apply (lambda row: bin2income (row),axis=1)

#Creat binary Alcohol consumption
def bin2alcohol(row):
   if row['alcconsumption'] <= 5 :
      return 0
   elif row['alcconsumption'] > 5 :
      return 1
#Apply the new variable bin2alcohol to the gapmind dataset
data['bin2alcohol'] = data.apply (lambda row: bin2alcohol (row),axis=1)

# create binary Female employee rate
def bin2femalemployee(row):
   if row['femaleemployrate'] <= 50 :
      return 0
   elif row['femaleemployrate'] > 50 :
      return 1
#Apply the new variable bin2alcohol to the gapmind dataset
data['bin2femalemployee'] = data.apply (lambda row: bin2femalemployee (row),axis=1)

##############################################################################
#                    LOGISTIC REGRESSION
##############################################################################
# logistic regression with binary breast cancer per 100th women
lreg1 = smf.logit(formula = 'bin2cancer ~ bin2alcohol',
                  data = data).fit()
print (lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (np.exp(lreg1.params))

# odd ratios with 95% confidence intervals
print ('Logistic regression with binary alcohol consumption')
print ('Odd ratios with 95% confidence intervals')
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))
print "\n-----------------------------\n"
# logistic regression with binary income per person and binary alcohol consumption
lreg2 = smf.logit(formula = 'bin2cancer ~ bin2income + bin2alcohol ',
                  data = data).fit()
print (lreg2.summary())

print ('\nLogistic regression with binary income per persone, binary alcohol consumption and binary female employee rate')
print ('\nOdd ratios with 95% confidence intervals')
# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

print "\n-----------------------------\n"
# logistic regression with binary income per person,binary alcohol consumption and female employee rate
lreg3 = smf.logit(formula = 'bin2cancer ~ bin2alcohol+bin2femalemployee',
                  data = data).fit()
print (lreg3.summary())

print ('\nLogistic regression with binary income per person, binary alcohol consumption and binary female employee rate')
print ('\nOdd ratios with 95% confidence intervals')
# odd ratios with 95% confidence intervals
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))