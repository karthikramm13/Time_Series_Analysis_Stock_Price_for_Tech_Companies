#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings 
filterwarnings('ignore')


# ### Tech stocks used for Analysis

# In[2]:


path = 'E:\Time Series Analysis'
company_list = ['EBAY_data.csv', 'FB_data.csv', 'GE_data.csv', 'IBM_data.csv']

#Blank DataFrame
all_data = pd.DataFrame()

for file in company_list:
    current_df = pd.read_csv(path + '/' + file)
    all_data = pd.concat([all_data, current_df])
    
all_data.shape


# In[3]:


all_data.head()


# In[4]:


all_data.dtypes


# In[5]:


all_data['date'] = pd.to_datetime(all_data['date'])


# In[6]:


all_data['date'][0]


# In[7]:


all_data.columns


# ### Analyse closing price of all the stocks

# In[8]:


tech_list=all_data['Name'].unique()


# In[9]:


tech_list


# In[10]:


plt.figure(figsize=(20,12))
for i, company in enumerate(tech_list,1):
    plt.subplot(2,2, i)
    df=all_data[all_data['Name']==company]
    plt.plot(df['date'], df['close'])
    plt.title(company)


# ### Analysing the total volume of stock being traded each day

# In[11]:


plt.figure(figsize=(20,12))
for i, company in enumerate(tech_list,1):
    plt.subplot(2,2, i)
    df=all_data[all_data['Name']==company]
    plt.plot(df['date'], df['volume'])
    plt.title(company)


# ### Using Plotly

# In[12]:


import plotly.express as px


# In[13]:


for company in (tech_list):
    df = all_data[all_data['Name']==company]
    fig = px.line(df, x='date', y='volume', title=company)
    fig.show()


# ### Analyse daily price change in stock
# #### To calculate how much you gained or lost per day for a stock, subtract the opening price from the closing price. Then, multiply the result by the number of shares you own in the company. 

# In[14]:


df = pd.read_csv("E:\Time Series Analysis/EBAY_data.csv")
df.head()


# ### Percentage return

# In[15]:


df['1day % return'] = ((df['close']-df['open'])/df['close'])*100
df.head()


# In[16]:


df.columns


# ### Using plotly to visualize

# In[17]:


import plotly.express as px
fig = px.line(df, x='date', y='1day % return')
fig.show()


# ### Using matplotlib for Vizualization

# In[18]:


plt.figure(figsize=(10,6))
df['1day % return'].plot()


# ### Plotting the intervals

# In[19]:


df.set_index('date')['2015-08-01':'2015-11-01']['1day % return'].plot()
plt.xticks(rotation='vertical')


# ### Analyse monthly mean of close column

# In[22]:


df2 = df.copy()


# In[23]:


df2['date'] = pd.to_datetime(df2['date'])


# In[24]:


df2.set_index('date', inplace=True)


# In[25]:


df2.head()


# In[26]:


df2['close'].resample('M').mean()  # Here we have assigned EBAY data only to 'df2' which is copied from 'df'


# In[29]:


df2['close'].resample('M').mean().plot()


# ### Resampling Close column year wise

# In[31]:


df2['close'].resample('Y').mean().plot()


# ### Finding the correlation between the stock prices of these Tech companies

# In[32]:


df.head()


# ### Reading datas of tech companies

# In[34]:


ebay = pd.read_csv("E:\Time Series Analysis/EBAY_data.csv")
ebay.head()


# In[35]:


fb = pd.read_csv("E:\Time Series Analysis/FB_data.csv")
fb.head()


# In[36]:


ge = pd.read_csv("E:\Time Series Analysis/GE_data.csv")
ge.head()


# In[37]:


ibm = pd.read_csv("E:\Time Series Analysis/IBM_data.csv")
ibm.head()


# In[38]:


# Creat a blank dataframe

close = pd.DataFrame()


# In[40]:


close['ebay'] = ebay['close']
close['fb'] = fb['close']
close['ge'] = ge['close']
close['ibm'] = ibm['close']


# In[41]:


close.head()


# ### Multivariante Analysis

# In[43]:


sns.pairplot(data=close)


# ### Correlation plot for stock prices

# In[45]:


sns.heatmap(close.corr(), annot=True)


# ### Analyse daily return of each stock and how they are correlated

# In[46]:


data = pd.DataFrame()


# In[47]:


ebay.head()


# In[49]:


data['ebay_change']=((ebay['close']-ebay['open'])/ebay['close'])*100
data['fb_change']=((fb['close']-fb['open'])/fb['close'])*100
data['ge_change']=((ge['close']-ge['open'])/ge['close'])*100
data['ibm_change']=((ibm['close']-ibm['open'])/ibm['close'])*100


# In[50]:


data.head()


# In[52]:


sns.pairplot(data=data)


# In[53]:


sns.heatmap(data.corr(), annot=True)


# ### Value at Risk Analysis for EBAY

# In[54]:


sns.distplot(data['ebay_change'])


# #### In the above plot, it somehow follows the Normal Distribution

# In[56]:


data['ebay_change'].std()


# In[57]:


data['ebay_change'].quantile(0.1)


# #### -1.4839329054190722 means that 90% of the times the worst daily Loss will not exceed 1.42

# In[60]:


data.describe().T


# In[ ]:




