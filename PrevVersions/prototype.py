#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
  monday = monday.strftime("%-m/%-d/%Y")
  tuesday = tuesday.strftime("%-m/%-d/%Y")
  wednesday = wednesday.strftime("%-m/%-d/%Y")
  thursday = thursday.strftime("%-m/%-d/%Y")
  friday = friday.strftime("%-m/%-d/%Y")

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


# In[ ]:


# read in the prices file with the pandas library to create a dataframe
prices = pd.read_csv('/roku-prices-18_19.csv', skiprows=1, skip_blank_lines=True)

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
prices.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)

# add a new column for the month number to the table
# month numbers are 1-24, January 2018 is month 1 and January 2019 is month 13
prices['Month Number'] = prices.apply(lambda row: get_month_number(row.Date), axis = 1) 
print(prices)


# In[ ]:


# read in the trends csv (weekly mentions)
trends = pd.read_csv('roku/roku-trends.csv', skiprows=1, skip_blank_lines=True)
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

#print(trends)


# trends now has initial Sunday dates, plus all the new, reformatted week days to match the formatting in the prices df


# In[ ]:


# combine the two dataframes (prices and trends) based on the unique ID (date), add the weekly mentions column to the existing prices
# df. create a new name for this combined df. 

trends = trends.rename(columns={"Week": "Date"})

prices_and_trends = pd.merge(left=prices, right=trends, left_on='Date', right_on='Date')

# prices_and_trends is a dataframe that contains the columns: Date, Close ($ value), Month Number (for randomizing training/test sets),
# and Weekly Mentions (every day has the weekly mentions for its respective week) 


# In[ ]:


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


# prices_trends_and_indicators is a dataframe that contains the important columns from each of the three csv files
# prices, trends, and indicators. The columns are: Date, Close ($ value), Month Number (for randomizing training/test sets),
# and Weekly Mentions (every day has the weekly mentions for its respective week), Volume, PB, PS, and MOA


# In[ ]:


# write the prices_trends_and_indicators dataframe to a csv file, just in case for future reference

prices_trends_and_indicators.to_csv(r'roku-prices-trends-indicators.csv', index = False)


# In[ ]:


# create the randomization of months (numbered 1-24) for the training and test set

order_arr = []
while len(order_arr) < 24:
  n = random.randint(1,24)
  if n not in order_arr:
    order_arr.append(n)

# split the randomly ordered array of month numbers into the training and test sets
# the training set takes 16 months of the data and the test set takes 8 months of the data
training_set = order_arr[0:16]
test_set = order_arr[16:]


# In[ ]:


# statistcal details of the dataset
prices_trends_and_indicators.describe()


# In[72]:


# carry out multivariate linear regression for training and predictive purposes

# separate the independent variables (in X) and dependent variable (in y)
X = prices_trends_and_indicators[['Weekly Mentions', 'Volume', 'PB', 'PS', 'MOA']]
y = prices_trends_and_indicators['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# to check the coefficients and intercept used for the independent variables
print("Coefficients: \n", regressor.coef_)
print("Intercept: \n", regressor.intercept_)


# In[73]:


y_pred = regressor.predict(X_test)

# check actual vs. predicted values
act_vs_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
act_vs_pred.head()


# In[74]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[75]:


# get a summary with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

