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
# FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric
#data['breastcancerper100th']= pd.to_numeric(data['breastcancerper100th'],errors='ignore')
data['breastcancerper100th']=data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['alcconsumption']= data['alcconsumption'].convert_objects(convert_numeric=True)
#data['lifeexpectancy']=data['lifeexpectancy'].convert_objects(convert_numeric=True)
data['femaleemployrate']=data['femaleemployrate'].convert_objects(convert_numeric=True)
data['lifeexpectancy']=data['lifeexpectancy'].convert_objects(convert_numeric=True)
data['urbanrate']=data['urbanrate'].convert_objects(convert_numeric=True)

# creating a copy of the dataframe
sub11=data.copy()
# Checking for missing values in the data frame
print "\n\t Missing values count\n",sub11.isnull().sum()

# Since the data is all continuous variables therefore the use the mean() for missing value imputation
sub11.fillna(sub11['breastcancerper100th'].mean(), inplace=True)
sub11.fillna(sub11['femaleemployrate'].mean(), inplace=True)
sub11.fillna(sub11['alcconsumption'].mean(), inplace=True)
sub11.fillna(sub11['lifeexpectancy'].mean(), inplace=True)
sub11.fillna(sub11['urbanrate'].mean(), inplace=True)
print "\nMissing value imputation done\n",sub11.isnull().sum()
# Now centering the quantative IV for regression analysis
sub11['alcconsumption_c']=(sub11['alcconsumption'] - sub11['alcconsumption'].mean())
sub11['femaleemployrate_c']=(sub11['femaleemployrate'] - sub11['femaleemployrate'].mean())
sub11['lifeexpectancy_c']=sub11['lifeexpectancy'] - sub11['lifeexpectancy'].mean()
sub11['urbanrate_c']=sub11['urbanrate'] - sub11['urbanrate'].mean()
# Now checking the centered IV
print "\n#### Centered quantative IVs ####"
print "\nAlcohol consumption= ",sub11['alcconsumption_c'].mean()
print "\nFemale employee rate= ",sub11['femaleemployrate_c'].mean()
print "\nLife Expectancy= ",sub11['lifeexpectancy_c'].mean()
print "\nUrban rate= ",sub11['urbanrate_c'].mean()


# Now checking for the association between breast cancer and female employee rate
reg11=smf.ols('breastcancerper100th~alcconsumption_c + femaleemployrate_c',data=sub11).fit()
print reg11.summary()

# Now checking for multiple variables signicance in regression analysis
reg12=smf.ols('breastcancerper100th~alcconsumption_c + femaleemployrate_c+lifeexpectancy_c',data=sub11).fit()
print reg12.summary()

# Now checking for multiple variables signicance in regression analysis
reg13=smf.ols('breastcancerper100th~alcconsumption_c + femaleemployrate_c+lifeexpectancy_c+urbanrate_c',data=sub11).fit()
print reg13.summary()


############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################
scat1 = seaborn.regplot(x="breastcancerper100th", y="alcconsumption", order=2, scatter=True, data=sub11)

plt.xlabel('Breast Cancer per 100th woman')
plt.ylabel ('Alcohol Consumption')
plt.title ('Scatterplot for the association between Breast Cancer & Alcohol Consumption')
#print scat1
plt.show()

print ("OLS regression model for the association between Breast Cancer & Alcohol Consumption")
reg14 = smf.ols('breastcancerper100th ~ alcconsumption', data=sub11).fit()
print (reg14.summary())

# regression model with second order polynomial
reg15=smf.ols('breastcancerper100th ~ alcconsumption_c+I(alcconsumption_c**2)',data=sub11).fit()
print reg15.summary()

################################################################################################
# Regression Diagnostic Plot
################################################################################################

# Q-Q plot for normality
fig1=sm.qqplot(reg15.resid,line='r')
plt.show()
#print fig1

# simple plot of residuals
stdres=pd.DataFrame(reg15.resid_pearson)
fig2=plt.plot(stdres,'o',ls='None')
l=plt.axhline(y=0,color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
plt.show()

# Leverage plot
fig3=sm.graphics.influence_plot(reg15,size=8)
plt.show()