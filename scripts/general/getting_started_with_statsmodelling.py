# Reference: http://www.statsmodels.org/stable/gettingstarted.html

# load the required libraries
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices as dm

# download the data
df = sm.datasets.get_rdataset("Guerry", "HistData").data
# select the vars of interest
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df[-5:]
df = df.dropna()
df[-5:]
