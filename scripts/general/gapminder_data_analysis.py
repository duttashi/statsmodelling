# Importing the required libraries
# Note %matplotlib inline works only for ipython notebook. It will not work for PyCharm. It is used to show the plot distributions
# Make sure to put %matplotlib inline as the first line of code when visualising plots. Also in pyCharm IDE use plt.show() to see the plot
%matplotlib inline 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
# Reading the data where low_memory=False increases the program efficiency
data= pd.read_csv("gapminder.csv", low_memory=False)
# setting variables that you will be working with to numeric
data['breastcancerper100th']= data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['femaleemployrate']= data['femaleemployrate'].convert_objects(convert_numeric=True)
data['alcconsumption']= data['alcconsumption'].convert_objects(convert_numeric=True)
# shows the number of rows and columns
print (len(data)) 
print (len(data.columns))
print (len(data.index))
# Print the column headers/headings
names=data.columns.values
print names
# using the describe function to get the standard deviation and other descriptive statistics of our variables
desc1=data['breastcancerper100th'].describe()
desc2=data['femaleemployrate'].describe()
desc3=data['alcconsumption'].describe()
print "\nBreast Cancer per 100th person\n", desc1
print "\nfemale employ rate\n", desc2
print "\nAlcohol consumption in litres\n", desc3

data.describe()
# Show the frequency distribution
print "\nAlcohol Consumption\nFrequency Distribution (in %)"
c1=data['alcconsumption'].value_counts(sort=False,dropna=False)
print c1
print "\nBreast Cancer per 100th"
c2=data['breastcancerper100th'].value_counts(sort=False)
print c2
print "\nFemale Employee Rate"
c3=data['femaleemployrate'].value_counts(sort=False)
print c3
# Show the frequency distribution of the quantitative variable using the groupby function
ac1=data.groupby('alcconsumption').size()
print "ac1\n",ac1
# Creating a subset of the data
sub1=data[(data['femaleemployrate']>40) & (data['alcconsumption']>=20)& (data['breastcancerper100th']<50)]
# creating a copy of the subset. This copy will be used for subsequent analysis
sub2=sub1.copy()
print "\nContries where Female Employee Rate is greater than 40 &" \
      " Alcohol Consumption is greater than 20L & new breast cancer cases reported are less than 50\n"
print sub2
print "\nContries where Female Employee Rate is greater than 50 &" \
      " Alcohol Consumption is greater than 10L & new breast cancer cases reported are greater than 70\n"
sub3=data[(data['alcconsumption']>10)&(data['breastcancerper100th']>70)&(data['femaleemployrate']>50)]
print sub3
# Checking for missing values in the data row-wise 

print "Missing data rows count: ",sum([True for idx,row in data.iterrows() if any(row.isnull())])
# Checking for missing values in the data column-wise

print "Showing missing data coulmn-wise"
print data.isnull().sum()
# Create a copy of the original dataset as sub4 by using the copy() method
sub4=data.copy()
# Now showing the count of null values in the variables
print sub4.isnull().sum()
# Since the data is all continuous variables therefore the use the mean() for missing value imputation 
# if dealing with categorical data, than use the mode() for missing value imputation

sub4.fillna(sub4['breastcancerper100th'].mean(), inplace=True)
sub4.fillna(sub4['femaleemployrate'].mean(), inplace=True)
sub4.fillna(sub4['alcconsumption'].mean(), inplace=True)
# Showing the count of null values after imputation
print sub4.isnull().sum()
# categorize quantitative variable based on customized splits using the cut function
sub4['alco']=pd.qcut(sub4.alcconsumption,6,labels=["0","1-4","5-9","10-14","15-19","20-24"])
sub4['brst']=pd.qcut(sub4.breastcancerper100th,5,labels=["1-20","21-40","41-60","61-80","81-90"])
sub4['emply']=pd.qcut(sub4.femaleemployrate,4,labels=["30-39","40-59","60-79","80-90"])

# Showing the frequency distribution of the categorised quantitative variables
print "Frequency distribution of the categorized quantitative variables\n"
fd1=sub4['alco'].value_counts(sort=False,dropna=False)
fd2=sub4['brst'].value_counts(sort=False,dropna=False)
fd3=sub4['emply'].value_counts(sort=False,dropna=False)
print "Alcohol Consumption\n",fd1
print "\n------------------------\n"
print "Breast Cancer per 100th\n",fd2
print "\n------------------------\n"
print "Female Employee Rate\n",fd3
print "\n------------------------\n"

# Now plotting the univariate quantitative variables using the distribution plot
sub5=sub4.copy()
sns.distplot(sub5['alcconsumption'].dropna(),kde=True)

plt.xlabel('Alcohol consumption in litres')
plt.title('Breast cancer in working class women')
plt.show() # Note: Although there is no need to use the show() method for ipython notebook as %matplotlib inline does the trick but 
           #I am adding it here because matplotlib inline does not work for an IDE like Pycharm and for that i need to use plt.show
sns.distplot(sub5['breastcancerper100th'].dropna(),kde=True)
plt.xlabel('Breast cancer per 100th women')
plt.title('Breast cancer in working class women')
plt.show()

sns.distplot(sub5['femaleemployrate'].dropna(),kde=True)
plt.xlabel('Female employee rate')
plt.title('Breast cancer in working class women')
plt.show()
# using scatter plot the visulaize quantitative variable. 
# if categorical variable then use histogram
scat1= sns.regplot(x='alcconsumption', y='breastcancerper100th', data=data)
plt.xlabel('Alcohol consumption in liters')
plt.ylabel('Breast cancer per 100th person')
plt.title('Scatterplot for the Association between Alcohol Consumption and Breast Cancer 100th person')

scat2= sns.regplot(x='femaleemployrate', y='breastcancerper100th', data=data)
plt.xlabel('Female Employ Rate')
plt.ylabel('Breast cancer per 100th person')
plt.title('Scatterplot for the Association between Female Employ Rate and Breast Cancer per 100th Rate')
