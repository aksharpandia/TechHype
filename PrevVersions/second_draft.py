#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import math
from datetime import datetime
from datetime import timedelta
import random
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm


# function to scan the date given and see what month number it should be assigned
# the numbering works so that January of 2018 is 1, April of 2018 is 4,
# January of 2019 is 13, and so forth
def get_month_number(date):
  month = date.split("/")[0]
  month_num = int(month)
  year = date.split("/")[2]
  year = int(year)
  if year == 2019:
    month_num += 12
  return month_num


# function that takes each Sunday date and calculates the dates for the 5 weekdays for the coming
# week. So if you have 1/7/2018, the output will be the following weekdays: 1/8/2018, 1/9/2018, 1/10/2018, 
# 1/11/2018, and 1/12/2018.
# ASSUMPTION: each week starts on Sunday and the input date is in the form year-month-day WITH DASHES
# ASSUMPTIONS CONT.: the format of the date for the prices dateframe is month/day/year WITH FORWARD SLASH
def get_weekdays(date_str):
  given_date = datetime.strptime(date_str, '%Y-%m-%d').date()
 # print(type(given_date))
 # print(given_date)
  monday = given_date + timedelta(days=1)
  tuesday = given_date + timedelta(days=2)
  wednesday = given_date + timedelta(days=3)
  thursday = given_date + timedelta(days=4)
  friday = given_date + timedelta(days=5)

  # reformat dates into the right format for the prices dataframe
  monday = '{0.month}/{0.day}/{0.year}'.format(monday)
  tuesday = '{0.month}/{0.day}/{0.year}'.format(tuesday)
  wednesday = '{0.month}/{0.day}/{0.year}'.format(wednesday)
  thursday = '{0.month}/{0.day}/{0.year}'.format(thursday)
  friday = '{0.month}/{0.day}/{0.year}'.format(friday)

  return[monday, tuesday, wednesday, thursday, friday]


# arranges the weekdays in an array of arrays with the format of 
# [[date, weekly mentions], [date, weekly mentions], ..., [date, weekly mentions]]
# index 1 should be the same for all the days of a given week.
def build_date_mention_arr(array_of_dates, mentions):
  m = array_of_dates[0]
  tu = array_of_dates[1]
  w = array_of_dates[2]
  th = array_of_dates[3]
  f = array_of_dates[4]

  date_mention_arr = [[m, mentions], [tu, mentions], [w, mentions], [th, mentions], [f, mentions]]
  return date_mention_arr


# In[2]:


# read in the prices file with the pandas library to create a dataframe
prices = pd.read_csv('roku-prices-18-19.csv', skiprows=1, skip_blank_lines=True)

# filter out rows with empty stock values. they randomly appear throughout both 2018 and 2019
a = 0
for ind in prices.index: 
    n = prices['Open'][ind]
    if (math.isnan(n))==True:
      prices.drop([a], inplace=True)
    a += 1

# reset indices for rows after deleting rows to get continguous numbers
prices.reset_index(drop=True, inplace=True)

# delete all columns except the date and close value
prices.drop(['High', 'Low', 'Volume'], axis=1, inplace=True)

# add a new column for the month number to the table
# month numbers are 1-24, January 2018 is month 1 and January 2019 is month 13
prices['Month Number'] = prices.apply(lambda row: get_month_number(row.Date), axis = 1) 
print(prices)


# In[3]:


# read in the trends csv (weekly mentions)
trends = pd.read_csv('roku-trends.csv', skiprows=1, skip_blank_lines=True)
trends = trends.rename(columns={"$roku: (United States)": "Weekly Mentions"})

#trends.head()

# change type of "Week" column to datetime type, to easily get the dates of the next 5 days
for ind in trends.index: 
    d = trends['Week'][ind]
    mentions = trends['Weekly Mentions'][ind]
    dates_arr = get_weekdays(d)
    #print(dates_arr)
    # dates_arr[0] == monday, dates_arr[1] == tuesday, etc.
    d_m_arr = build_date_mention_arr(dates_arr, mentions)
    #print(d_m_arr)

    mini_trends_df = pd.DataFrame(d_m_arr, columns=['Week', 'Weekly Mentions'])
    #print(mini_trends_df)
    trends = trends.append(mini_trends_df, ignore_index=True)

print(trends)


# trends now has initial Sunday dates, plus all the new, reformatted week days to match the formatting in the prices df


# In[4]:


# combine the two dataframes (prices and trends) based on the unique ID (date), add the weekly mentions column to the existing prices
# df. create a new name for this combined df. 

trends = trends.rename(columns={"Week": "Date"})

prices_and_trends = pd.merge(left=prices, right=trends, left_on='Date', right_on='Date')

# prices_and_trends is a dataframe that contains the columns: Date, Close ($ value), Month Number (for randomizing training/test sets),
# and Weekly Mentions (every day has the weekly mentions for its respective week) 


# In[5]:


# reading in the indicators csv to create a new df

indicators = pd.read_csv('roku-indicators.csv', skiprows=1, skip_blank_lines=True)

# filter out rows with empty indicator values. they randomly appear throughout the file
c = 0
for ind in indicators.index: 
    n = indicators['Volume'][ind]
    if (math.isnan(n))==True:
      indicators.drop([c], inplace=True)
    c += 1


prices_trends_and_indicators = pd.merge(left=prices_and_trends, right=indicators, left_on='Date', right_on='Date')
prices_trends_and_indicators['price change'] = np.nan
start = 0
for i in range(1,len(prices_trends_and_indicators['price change'])):
    open_price = prices_trends_and_indicators['Open'][i]
    close_price = prices_trends_and_indicators['Close'][i]
    prices_trends_and_indicators['price change'][start] = float((close_price-open_price)/open_price)
    start+=1

indicators_to_shift = ['Weekly Mentions', 'Volume', 'PB', 'PS', 'MOA']
for indi in indicators_to_shift:
    shifted_indicator = prices_trends_and_indicators[indi][:-1]
    shifted_indicator_title = 'Shifted' + indi
    prices_trends_and_indicators[shifted_indicator_title]= shifted_indicator.astype(float)
# prices_trends_and_indicators is a dataframe that contains the important columns from each of the three csv files
# prices, trends, and indicators. The columns are: Date, Close ($ value), Month Number (for randomizing training/test sets),
# and Weekly Mentions (every day has the weekly mentions for its respective week), Volume, PB, PS, and MOA


# In[11]:


# write the prices_trends_and_indicators dataframe to a csv file, just in case for future reference

prices_trends_and_indicators.to_csv('roku-prices-trends-indicators-fin.csv', index = False)
prices_trends_and_indicators.head()
print(len(prices_trends_and_indicators))


# In[12]:


# create the randomization of months (numbered 1-24) for the training and test set

order_arr = []
while len(order_arr) < 17:
  n = random.randint(1,17)
  if n not in order_arr:
    order_arr.append(n)

# split the randomly ordered array of month numbers into the training and test sets
# the training set takes 16 months of the data and the test set takes 8 months of the data
training_set = order_arr[0:11]
test_set = order_arr[11:]
print(training_set)
print(test_set)


# In[13]:


# statistcal details of the dataset
prices_trends_and_indicators.describe()


# In[14]:


shifted_df = prices_trends_and_indicators[['price change','ShiftedWeekly Mentions', 'ShiftedVolume', 'ShiftedPB', 'ShiftedPS', 'ShiftedMOA']].copy()
remove_na_shifted = shifted_df.dropna()
print(len(shifted_df.dropna()))
print(remove_na_shifted.head())


# In[33]:


# carry out multivariate linear regression for training and predictive purposes

# separate the independent variables (in X) and dependent variable (in y)
X = remove_na_shifted[['ShiftedWeekly Mentions', 'ShiftedVolume', 'ShiftedPB', 'ShiftedPS', 'ShiftedMOA']]
y = remove_na_shifted['price change']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# carry out multivariate linear regression for training and predictive purposes

# separate the independent variables (in X) and dependent variable (in y)
# X = prices_trends_and_indicators[['Weekly Mentions', 'Volume', 'PB', 'PS', 'MOA']]
# y = prices_trends_and_indicators['Close']

x1 = X[0:30]
x2 = X[30:60]
x3 = X[60:90]
x4 = X[90:120]
x5 = X[120:150]
x6 = X[150:180]
x7 = X[180:210]
x8 = X[210:240]
x9 = X[240:270]
x10 = X[270:300]
x11 = X[300:330]
x12 = X[330:360]
x13 = X[360:390]
x14 = X[390:420]
x15 = X[420:450]
x16 = X[450:480]
x17 = X[480:499]

y1 = y[0:30]
y2 = y[30:60]
y3 = y[60:90]
y4 = y[90:120]
y5 = y[120:150]
y6 = y[150:180]
y7 = y[180:210]
y8 = y[210:240]
y9 = y[240:270]
y10 = y[270:300]
y11 = y[300:330]
y12 = y[330:360]
y13 = y[360:390]
y14 = y[390:420]
y15 = y[420:450]
y16 = y[450:480]
y17 = y[480:499]

dx={1:x1, 2:x2, 3:x3, 4:x4, 5:x5, 6:x6, 7:x7, 8:x8, 9:x9, 10:x10, 11:x11, 12:x12, 13:x13, 14:x14, 15:x15, 16:x16, 17:x17}
pd.concat(dx)

dy={1:y1, 2:y2, 3:y3, 4:y4, 5:y5, 6:y6, 7:y7, 8:y8, 9:y9, 10:y10, 11:y11, 12:y12, 13:y13, 14:y14, 15:y15, 16:y16, 17:y17}
pd.concat(dy)
X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()
for i in training_set:
    x_train_temp = dx[i]
    y_train_temp = dy[i]
    X_train = pd.concat([X_train, x_train_temp], ignore_index=False)
    y_train = pd.concat([y_train, y_train_temp], ignore_index=False)
    
    
for j in test_set:
    x_test_temp = dx[j]
    y_test_temp = dy[j]
    X_test = pd.concat([X_test, x_test_temp], ignore_index=False)
    y_test = pd.concat([y_test, y_test_temp], ignore_index=False)


# regressor = LinearRegression()  
# regressor.fit(X_train, y_train)

# # to check the coefficients and intercept used for the independent variables
# print("Coefficients: \n", regressor.coef_)
# print("Intercept: \n", regressor.intercept_)
# print(len(X_train))
# X_train.fillna(X_train.mean(), inplace=True)
# print(X_train)

# for item in X_train['ShiftedVolume']:
#     if item==np.nan:
#         print(0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# to check the coefficients and intercept used for the independent variables
print("Coefficients: \n", regressor.coef_)
print("Intercept: \n", regressor.intercept_)


# In[34]:


y_pred = regressor.predict(X_test)
y_new_test = y_test.to_numpy()

# check actual vs. predicted values

# act_vs_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# act_vs_pred.head()


# In[35]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[36]:


# get a summary with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# In[ ]:




