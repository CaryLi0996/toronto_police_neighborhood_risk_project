#!/usr/bin/env python
# coding: utf-8

# ## Organization Overview: 
# The Toronto Police Service (TPS) is a municipal police force in Toronto, Ontario, Canada. It is the largest municipal police service in Canada, and third largest police force in Canada after the Ontario Provincial Police (OPP) and the Royal Canadian Mounted Police (RCMP).
# 

# ## Datasets Overview:
# 
# * Toronto Police KSI (Killed/Seriously Injured): Identify when, how and where most impactful Killed and Seriously Injured accidents occur to reduce the incidents in neighbourhoods. 
# 
# * Toronto Police MCI (Major Crime Indicator): Help Police forces to identify occurrence of MCI based on area, time of day, weekday so that Police Patrols can be delegated accordingly. 
# 

# ## Descriptive Analysis
# 
# #### We tried to answer below questions and make some predictions after analysing it
# * 1.  Total number of KSI accidents in the City of Toronto in percentage
# * 2.  Total number of different crime types in the City of Toronto in percentage 
# * 3.  Trend Visualization for all crimes and KSI accidents by year.
# * 4.  What time of the day has the most accidents involved - Daylight, Early Eve, Late Eve, Night - added new attribute
# * 5.  Visualization of Location by neighbourhood heat map on both KSI and MSI dataset
# 
# ## AI Solution
# 
# * 6.  Time Seris Forecasting - Forecast next year general trend (Yearly,Monthly)
# * 7.  Clustering Neighbourhoods Risk Level

# ## Import Packages

# In[2]:


import numpy as np
import pandas as pd

from pandas.plotting import autocorrelation_plot, scatter_matrix

#visualization 
import matplotlib.pyplot as plt
import seaborn as sea

from pandas import DataFrame, Series
import statsmodels.formula.api as sm

import scipy, scipy.stats

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import statsmodels.api as sm

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

# Importing everything from forecasting quality metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import time


# ### Import Datasets

# In[ ]:


mci_df= pd.read_csv('../input/mcitoronto/MCI_2014_to_2019.csv')
ksi_df = pd.read_csv('../input/ksi-toronto/Motor Vehicle Collisions with KSI Data.csv')


# #### Check MCI dataset

# In[4]:



mci_df.head()


# #### Check KSI dataset

# In[5]:


ksi_df.head()


# ### Cleaning data

# In[6]:


# Drop Na
mci_df=mci_df.dropna()


# In[7]:


# Filter columns to be used
df_MCI=mci_df[['Hood_ID','Division','MCI','occurrencedate','occurrencehour']]
df_MCI['occurrencedate']=pd.to_datetime(df_MCI['occurrencedate']).dt.date
# Filter year
df_MCI=df_MCI.loc[pd.to_datetime(df_MCI['occurrencedate']).dt.year>=2014]


# In[8]:


#Combine same type in the same hour
df_MCI_count=df_MCI
df_MCI_count["Count"] = 1
df_MCI_count=df_MCI_count.groupby(['Hood_ID','Division','MCI','occurrencedate','occurrencehour']).count()


# In[9]:


#Sort by Date
df_MCI_count=df_MCI_count.sort_values(['occurrencedate','occurrencehour']).reset_index()
#Rename
df_MCI_count=df_MCI_count.rename(columns={"MCI": "Type", "occurrencedate": "Date","occurrencehour": "Hour"})


# In[10]:


#Finish MCI dataset
df_MCI_count.head()


# In[11]:


#Select KSI columns
df_Accident=ksi_df[['Hood_ID','Division','INJURY','DATE','HOUR']]


# In[12]:


#Rename to match MCI dataset , Seperate DATE to Month and Day Columns  and 

df_Accident=df_Accident.rename(columns={"INJURY": "Type", "DATE": "Date","HOUR": "Hour"})

df_Accident['Date']=pd.to_datetime(df_Accident['Date']).dt.date


#Only watch 2014 +
df_Accident=df_Accident.loc[pd.to_datetime(df_Accident['Date']).dt.year>=2014]


# In[13]:


df_Accident["Count"] = 1
df_Accident=df_Accident.groupby(['Hood_ID','Division','Type','Date','Hour']).count().reset_index()
df_Accident.head()


# In[14]:


#
df_Accident['Type']=df_Accident['Type']+' Collision'


# In[15]:


frames = [df_MCI_count, df_Accident]
df_All= pd.concat(frames)
df_All=df_All.sort_values(by='Date').reset_index(drop=True)


# In[16]:


df_All['Year'] = pd.to_datetime(df_All['Date']).dt.year
df_All['Month'] = pd.to_datetime(df_All['Date']).dt.month
df_All['Day']= pd.to_datetime(df_All['Date']).dt.day


# In[17]:


df_All.head()


# In[18]:


# output for csv for further investgation
df_All.to_csv("./output.csv")


# ## Analysis 1.Total number of KSI accidents in the City of Toronto in percentage

# In[19]:


df_Accident['Year'] = pd.to_datetime(df_Accident['Date']).dt.year
df_Accident['Month'] = pd.to_datetime(df_Accident['Date']).dt.month
df_Accident['Day']= pd.to_datetime(df_Accident['Date']).dt.day
df_Accident.head()


# In[20]:


pivot_KSI=df_Accident.pivot_table(index=['Year','Type'],values='Count',aggfunc=np.sum)
pivot_KSI


# In[21]:


years = df_Accident['Year'].unique()
for year in years:
    y = pivot_KSI.iloc[pivot_KSI.index.get_level_values('Year') == year]['Count']
    total = np.sum(y)
    plt.pie(y, labels = df_Accident['Type'].unique(),autopct='%1.2f%%', startangle=90 )
    plt.title("Total number of KSI Collision in the City of Toronto in percentage in " + str(year) + " : " + str(total))
    plt.show() 


# In[22]:


plt.figure(figsize=(12,5))
plt.title = ('Collisions by Year')
sea.barplot(x="Year", y="Count",hue='Type',data=pivot_KSI.reset_index())
plt.show()


# ## Analysis 2. Total number of different crime types in the City of Toronto in percentage

# In[23]:


df_MCI_count['Year'] = pd.to_datetime(df_MCI_count['Date']).dt.year
df_MCI_count['Month'] = pd.to_datetime(df_MCI_count['Date']).dt.month
df_MCI_count['Day']= pd.to_datetime(df_MCI_count['Date']).dt.day
df_MCI_count.head()


# In[24]:


pivot_MCI=df_MCI_count.pivot_table(index=['Year','Type'],values='Count',aggfunc=np.sum)
pivot_MCI


# In[25]:


crime_type=['Assault','Auto Theft','Break and Enter','Robbery','Theft Over']
years = df_MCI_count['Year'].unique()
for year in years:
    y = pivot_MCI.iloc[pivot_KSI.index.get_level_values('Year') == year]['Count']
    total = np.sum(y)
    plt.pie(y, labels = crime_type,autopct='%1.2f%%', startangle=90 )
    plt.title("Total number of Crimes in the City of Toronto in percentage in " + str(year) + " : " + str(total))
    plt.show() 


# In[26]:


plt.figure(figsize=(12,5))
plt.title = ('Crimes by Year')
sea.barplot(x="Year", y="Count",hue='Type',data=pivot_MCI.reset_index())
plt.show()


# ## Analysis 3. Trend Visualization for all crimes and KSI accidents by year.

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 15]
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Year', ylabel='Count', dpi=100):
    plt.figure(figsize=(6,3), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    
for i in (df_All['Type'].unique()):
    df=df_All.loc[df_All['Type']==i]
    plot_df(df_All, x=df_All.Year.unique(), y=df.groupby('Year')['Count'].agg('sum'), title=i) 


# ## Analysis 4: What time of the day has the most accidents/crimes involved - Daylight, Early Eve, Late Eve, Night - added new attribute

# - First we will define the new attribute "TIMEOFDAY". We define labels or buckets as 
#  - 12AM-4AM - [00 to 4 hours]
#  - 4AM-8AM - [4 to 8 hours]
#  - 8AM-12PM - [8 to 12 hours]
#  - 12PM-4PM - [12 to 16 hours]
#  - 4PM-8PM - [16 to 20 hours]
#  - 8PM-12PM - [20 to Midnight]

# In[28]:


bins = [0, 4, 8, 12, 16, 20, np.inf]
labels = ['12AM-4AM', '4AM-8AM','8AM-12PM', '12PM-4PM', '4PM-8PM', '8PM-12PM']


# ### KSI accidents

# In[29]:


df_Accident["TIMEOFDAY"] = pd.cut(df_Accident["Hour"], bins, labels = labels)
df_Accident.groupby('TIMEOFDAY')['Count'].agg('sum')


# In[30]:


df_Accident_time = pd.DataFrame(df_Accident.groupby(['TIMEOFDAY','Type'])['Count'].agg('sum'))
df_Accident_time


# ### Visualization

# In[31]:



plt.figure(figsize=(12,5))
plt.title = ('Time of the day for accidents')
sea.barplot(x="TIMEOFDAY", y="Count",hue='Type',data=df_Accident_time.reset_index())

plt.show()


# ### Analysis

# - It is clear that most of the accidents occured during hours start from 4PM to 8PM, which is the time when people try to reach home after work. 
# - Another point to be noted here is, 12PM to 4PM has the second highest accidents, around lunch hours to afternoon. 
# - Most of the accidents occured in Daylight from 8AM to 8PM, which is mainly office hours. 

# ### MCI crimes

# In[32]:


df_MCI_count["TIMEOFDAY"] = pd.cut(df_MCI_count["Hour"], bins, labels = labels)
df_MCI_count.groupby('TIMEOFDAY')['Count'].agg('sum')


# In[33]:


df_MCI_time = pd.DataFrame(df_MCI_count.groupby(['TIMEOFDAY','Type'])['Count'].agg('sum'))
df_MCI_time


# ### Visulization

# In[34]:


plt.figure(figsize=(12,5))

sea.barplot(x="TIMEOFDAY", y="Count",hue='Type',data=df_MCI_time.reset_index())
plt.title = ('Time of the day for crimes')
plt.show()


# ### Analysis
# 
# - It is clear that most of the crimes occured during hours start from 4PM to 8PM, which is the time when people try to reach home after work. ,Most of them are Assualt
# - Another point to be noted here is, 4AM-8AM is the least crimes time period.
# - Most of the Break and Enter happened from 12 AM to 4 AM, which is midnight

# ## Analysis 5. Visualization of Location by neighbourhood heat map on both KSI and MSI dataset

# In[35]:


import geopandas as gpd
sns.set(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


regions = gpd.read_file('../input/folder/forAnalysis/Neighbourhoods/Neighbourhoods.shp')

regions['neighbourhood'] = regions['FIELD_7'].str.replace(' \(.+\)', '').str.lower()
regions.sample(5)


# In[37]:


df_Accident_Neighbourhood = df_Accident.groupby(['Hood_ID'])['Count'].agg('sum')
df_Accident_Neighbourhood.sort_values(ascending=False).head(10)


# In[38]:


merged = regions.set_index('FIELD_5').join(df_Accident_Neighbourhood)
merged = merged.reset_index()
merged = merged.fillna(0)
merged[['FIELD_7', 'FIELD_11', 'FIELD_12', 'geometry', 'Count']].sample(5)


# In[39]:


# we are using the maximum and minimum count values from the previous cell.
# setting additionally properties for the plot such as titles, turning of the axis for better visibility
# and setting the color scheme to look like a heat map.
fig, ax = plt.subplots(1, figsize=(20, 10))
ax.axis('off')
ax.set_title('Heat Map of KSI collisons in Toronto, Ontario', fontdict={'fontsize': '40', 'fontweight' : '3'})


# Create colorbar as a legend
# empty array for the data range
# add the colorbar to the figure
# set the color bar label text size
color = 'Oranges'
vmin, vmax = 0, 200
sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
cbar.ax.tick_params(labelsize=20)


# actually plot the map
merged.plot('Count', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
for idx, row in merged.iterrows():
    if(row['Count'] > 140):
        plt.annotate(s=row['FIELD_7'], xy=(row['FIELD_11'], row['FIELD_12']),
                 horizontalalignment='center', fontsize='large', color='black', wrap=True)
plt.show()


# In[40]:


df_MCI_Neighbourhood = df_MCI_count.groupby(['Hood_ID'])['Count'].agg('sum')
df_MCI_Neighbourhood.sort_values(ascending=False).head(10)


# In[41]:


MCI_merged = regions.set_index('FIELD_5').join(df_MCI_Neighbourhood)
MCI_merged = MCI_merged.reset_index()
MCI_merged = MCI_merged.fillna(0)
MCI_merged[['FIELD_7', 'FIELD_11', 'FIELD_12', 'geometry', 'Count']].sample(5)


# In[42]:


# we are using the maximum and minimum count values from the previous cell.
# setting additionally properties for the plot such as titles, turning of the axis for better visibility
# and setting the color scheme to look like a heat map.
fig, ax = plt.subplots(1, figsize=(20, 10))
ax.axis('off')
ax.set_title('Heat Map of Crimes in Toronto, Ontario', fontdict={'fontsize': '40', 'fontweight' : '3'})


# Create colorbar as a legend
# empty array for the data range
# add the colorbar to the figure
# set the color bar label text size
color = 'Blues'
vmin, vmax = 0,8000
sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
cbar.ax.tick_params(labelsize=20)


# actually plot the map
MCI_merged.plot('Count', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
for idx, row in MCI_merged.iterrows():
    if(row['Count'] > 6000):
        plt.annotate(s=row['FIELD_7'], xy=(row['FIELD_11'], row['FIELD_12']),
                 horizontalalignment='center', fontsize='large', color='black', wrap=True)
plt.show()


# ## AI Solution 6. TIME SERIES FORECASTING - Forecast next year general trend (Yearly,Monthly)
# 

# In[43]:


import warnings                                  # do not disturbe mode
warnings.filterwarnings('ignore')

# Load packages
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
from datetime import datetime

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

# Importing everything from forecasting quality metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

data = pd.read_csv('../input/newoutput/output (1).csv')

data['Time'] = pd.to_datetime(data['Month'])

df = data.groupby(['Time']).sum()


# In[44]:


df_2 = data.groupby(['Time', 'Type']).sum()
def create_sub_df(type_of_crime):
  return df_2[np.in1d(df_2.index.get_level_values(1), type_of_crime)]
assault = create_sub_df('Assault')
assault.index = assault.index.droplevel(1)
auto_theft = create_sub_df('Auto Theft')
auto_theft.index = auto_theft.index.droplevel(1)
break_and_enter = create_sub_df('Break and Enter')
break_and_enter.index = break_and_enter.index.droplevel(1)
fatal_collision = create_sub_df('Fatal Collision')
fatal_collision.index = fatal_collision.index.droplevel(1)
major_collision = create_sub_df('Major Collision')
major_collision.index = major_collision.index.droplevel(1)
minimal_collision = create_sub_df('Minimal Collision')
minimal_collision.index = minimal_collision.index.droplevel(1)
minor_collision = create_sub_df('Minor Collision')
minor_collision.index = minor_collision.index.droplevel(1)
none_collision = create_sub_df('None Collision')
none_collision.index = none_collision.index.droplevel(1)
robbery = create_sub_df('Robbery')
robbery.index = robbery.index.droplevel(1)
thetf_over = create_sub_df('Theft Over')
thetf_over.index = thetf_over.index.droplevel(1)


# In[45]:


# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[46]:


tsplot(df.Count, lags=50)


# In[47]:


ads_diff = df.Count - df.Count.shift(12)
tsplot(ads_diff[12:], lags=50)


# In[48]:



# setting initial values and some bounds for them
ps = range(2, 5)
d = 0 
qs = range(2, 5)
Ps = range(0, 2)
D = 1 
Qs = range(0, 2)
s = 12 # season length is still 12

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[49]:



def optimizeSARIMA(parameters_list, d, D, s):
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(df.Count, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table


# In[50]:


result_table = optimizeSARIMA(parameters_list, d, D, s)


# In[51]:


p, q, P, Q = result_table.parameters[0]
model_total=sm.tsa.statespace.SARIMAX(df.Count, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)


# In[52]:


ss = df.Count
ss.columns = ['actual']
ss


# In[53]:


def plotSARIMA(series, model, n_steps):
    # adding model values
    data = series.copy()
    data['actual'] = data[:]
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model due to the differentiating
    data['sarima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward
    forecast = model.predict(start = data.shape[0]-2, end = data.shape[0]+n_steps-2)
    forecast = data.sarima_model.append(forecast)
    
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['sarima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    #plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    #plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label='actual')
    plt.legend()
    plt.show()
    return(forecast)


# In[54]:


# Total Prediction
prediction_total = plotSARIMA(df.Count, model_total, 11)
prediction_total


# In[55]:


print('Total Crimes prediction for 2020')
prediction_total['2020-03-01']+prediction_total['2020-04-01']+prediction_total['2020-05-01']+prediction_total['2020-06-01']+prediction_total['2020-07-01']+prediction_total['2020-08-01']+prediction_total['2020-09-01']+prediction_total['2020-10-01']+prediction_total['2020-11-01']+prediction_total['2020-12-01']+prediction_total['2020-01-01']+prediction_total['2020-02-01']


# In[56]:


# Assault Prediction
model_assault=sm.tsa.statespace.SARIMAX(assault.Count, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
prediction_assault = plotSARIMA(assault.Count, model_assault, 11)
prediction_assault


# In[57]:


# Auto Theft Prediction
model_auto_theft=sm.tsa.statespace.SARIMAX(auto_theft.Count, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
prediction_auto_theft = plotSARIMA(auto_theft.Count, model_auto_theft, 11)
prediction_auto_theft


# In[58]:


# Break and Enter Prediction
model_break_and_enter=sm.tsa.statespace.SARIMAX(break_and_enter.Count, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
prediction_break_and_enter = plotSARIMA(break_and_enter.Count, model_break_and_enter, 11)
prediction_break_and_enter


# In[59]:


# Fatal Collision Prediction
model_fatal_collision=sm.tsa.statespace.SARIMAX(fatal_collision.Count, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
prediction_fatal_collision = plotSARIMA(fatal_collision.Count, model_fatal_collision, 11)
prediction_fatal_collision


# In[60]:


# Major Collision Prediction
model_major_collision=sm.tsa.statespace.SARIMAX(major_collision.Count, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
prediction_major_collision = plotSARIMA(major_collision.Count, model_major_collision, 11)
prediction_major_collision


# # AI Solution 7. Clustering Neighbourhoods

# In[61]:


df_All["TIMEOFDAY"] = pd.cut(df_All["Hour"], bins, labels = labels)
df_All_time = pd.DataFrame(df_All.groupby(['Hood_ID','TIMEOFDAY','Type'])['Count'].agg('sum'))
df_All_time.head()


# In[62]:


df_neighbourhoods = df_All_time.pivot_table('Count', ['Hood_ID'], ['Type'],aggfunc=np.sum)


# In[63]:


df_neighbourhoods=df_neighbourhoods.fillna(0)


# In[64]:


df_neighbourhoods.head()


# In[65]:


df_neighbourhoods = regions.set_index('FIELD_5')[['FIELD_7']].join(df_neighbourhoods)


# In[66]:


df_neighbourhoods.set_index('FIELD_7', inplace=True)


# In[67]:


df_neighbourhoods.head()


# In[68]:


from sklearn import preprocessing
df_neighbourhoods = df_neighbourhoods.apply(lambda x: x.astype('float64'))
df_norm = df_neighbourhoods.apply(preprocessing.scale, axis=0)


# In[69]:


from scipy.cluster.hierarchy import dendrogram, linkage
#Average
Z = linkage(df_norm, method='average')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[70]:


#Single
Z = linkage(df_norm, method='single')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[71]:


#Ward
Z = linkage(df_norm, method='ward')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[72]:


#Complete
Z = linkage(df_norm, method='complete')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[73]:


#Median
Z = linkage(df_norm, method='median')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[74]:


#Weighted
Z = linkage(df_norm, method='weighted')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[75]:


#Centroid
Z = linkage(df_norm, method='centroid')
plt.figure(figsize=(60,20))
fig.subplots_adjust(right=3)
plt.xlabel('HoodID')
dendrogram(Z, labels=df_norm.index, color_threshold=5.5)
plt.axhline(y=5.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[76]:


from sklearn.cluster import KMeans

# Fit a k-Means clustering with k=6 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_norm)

# Cluster membership
memb = pd.Series(kmeans.labels_, index=df_norm.index)
print('\033[1m'+'k-Means cluster membership:'+'\033[0m')
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[77]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df_norm.columns)
pd.set_option('precision', 3) # round to 3 decimal places
print(centroids)


# In[78]:


from pandas.plotting import parallel_coordinates
centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]

plt.figure(figsize=(30,6))
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.915, 0.5))
plt.xlim(-0.5,10)


# High Risk Neighbourhoods: Cluster 1
# * York University Heights (27)
# * Moss Park (73)
# * Waterfront Communities-The Island (77)
# * West Humber-Clairville (1)
# * Wexford/Maryvale (119)
# * Woburn (137)
# * Bay Street Corridor (76)
# * Church-Yonge Corridor (75)
# * Downsview-Roding-CFB (26)
# * Islington-City Centre West (14)
# 
# 
# 
# Medium Risk Neighbourhoods: Cluster 0
# 
# *  Yorkdale-Glen Park (31), Malvern (132), Milliken (130), Mimico (includes Humber Bay Shores) (17), Mount Olive-Silverstone-Jamestown (2), Newtonbrook East (50), Niagara (82), Rockcliffe-Smythe (111), Rouge (131), South Parkdale (85), South Riverdale (70), Steeles (116), Tam O'Shanter-Sullivan (118), West Hill (136), Willowdale East (51), Agincourt North (129), Agincourt South-Malvern West (128), Annex (95), Banbury-Don Mills (42), Bedford Park-Nortown (39), Bendale (127), Birchcliffe-Cliffside (122), Clairlea-Birchmount (120), Don Valley Village (47), Dorset Park (126), Dovercourt-Wallace Emerson-Junction (93), Eglinton East (138), Glenfield-Jane Heights (25), High Park-Swansea (87), Humber Summit (21), Humbermede (22), Junction Area (90), Kennedy Park (124), Kensington-Chinatown (78), L'Amoreaux (117)
# 
# 
# Low Risk Neighbourhoods: Cluster 2
# 
# * Wychwood (94), Yonge-Eglinton (100), Yonge-St.Clair (97), Lambton Baby Point (114), Lansing-Westgate (38), Lawrence Park North (105), Lawrence Park South (103), Leaside-Bennington (56), Little Portugal (84), Long Branch (19), Maple Leaf (29), Markland Wood (12), Morningside (135), Mount Dennis (115), Mount Pleasant East (99), Mount Pleasant West (104), New Toronto (18), Newtonbrook West (36), North Riverdale (68), North St.James Town (74), O'Connor-Parkview (54), Oakridge (121), Oakwood Village (107), Old East York (58), Palmerston-Little Italy (80), Parkwoods-Donalda (45), Pelmo Park-Humberlea (23), Playter Estates-Danforth (67), Pleasant View (46), Princess-Rosethorn (10), Regent Park (72), Rexdale-Kipling (4), Roncesvalles (86), Rosedale-Moore Park (98), Runnymede-Bloor West Village (89), Rustic (28), Scarborough Village (139), St.Andrew-Windfields (40), Stonegate-Queensway (16), Taylor-Massey (61), The Beaches (63), Thistletown-Beaumond Heights (3), Thorncliffe Park (55), Trinity-Bellwoods (81), University (79), Victoria Village (43), Westminster-Branson (35), Weston (113), Weston-Pellam Park (91), Willowdale West (37), Willowridge-Martingrove-Richview (7), Woodbine Corridor (64), Woodbine-Lumsden (60), Alderwood (20), Bathurst Manor (34), Bayview Village (52), Bayview Woods-Steeles (49), Beechborough-Greenbrook (112), Black Creek (24), Blake-Jones (69), Briar Hill-Belgravia (108), Bridle Path-Sunnybrook-York Mills (41), Broadview North (57), Brookhaven-Amesbury (30), Cabbagetown-South St.James Town (71), Caledonia-Fairbank (109), Casa Loma (96), Centennial Scarborough (133), Clanton Park (33), Cliffcrest (123), Corso Italia-Davenport (92), Danforth (66), Danforth East York (59), Dufferin Grove (83), East End-Danforth (62), Edenbridge-Humber Valley (9), Elms-Old Rexdale (5), Englemount-Lawrence (32), Eringate-Centennial-West Deane (11), Etobicoke West Mall (13), Flemingdon Park (44), Forest Hill North (102), Forest Hill South (101), Greenwood-Coxwell (65), Guildwood (140), Henry Farm (53), High Park North (88), Highland Creek (134), Hillcrest Village (48), Humber Heights-Westmount (8), Humewood-Cedarvale (106), Ionview (125), Keelesdale-Eglinton West (110), Kingsview Village-The Westway (6), Kingsway South (15)
# 
# 

# In[79]:


memb


# In[80]:


df_memb=memb
df_memb=pd.DataFrame(df_memb)

df_memb['cluster'] = memb


# In[81]:


risk_merged = regions.set_index('FIELD_7').join(df_memb)
risk_merged = risk_merged.reset_index()
risk_merged = risk_merged.fillna(0)
risk_merged.sample(5)


# In[82]:


def risk_func(x):
    return {
        2: 0,
        0: 1,
        1: 2
    }[x]
    
risk_merged['risk']=risk_merged['cluster'].map(risk_func)


# In[83]:


# we are using the maximum and minimum count values from the previous cell.
# setting additionally properties for the plot such as titles, turning of the axis for better visibility
# and setting the color scheme to look like a heat map.
fig, ax = plt.subplots(1, figsize=(30, 15))
ax.axis('off')
ax.set_title('Risk Level for Neighbourhoods in Toronto, Ontario', fontdict={'fontsize': '40', 'fontweight' : '3'})


# Create colorbar as a legend
# empty array for the data range
# add the colorbar to the figure
# set the color bar label text size
color = 'Oranges'
vmin, vmax = 0,3
sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
cbar.ax.tick_params(labelsize=20)


# actually plot the map
risk_merged.plot('risk', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
for idx, row in risk_merged.iterrows():
    if(row['risk'] > 1):
        plt.annotate(s=row['FIELD_7'], xy=(row['FIELD_11'], row['FIELD_12']),
                 horizontalalignment='center', fontsize='large', color='red', wrap=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




