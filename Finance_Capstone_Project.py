from pandas_datareader import data as web
import pandas as pd
import numpy as np
import datetime

#We need to get data using pandas-datareader
#We will get stock information for the following banks


#Getting the information on these stocks from
#01/2006 - 01/2016
#Bank of America
#CitiGroup
#Goldman Sachs
#JPMorgan Chase
#Morgan Stanley
#Wells Fargo

#Pandas-datareader allows you to read the stock
#info. directly from the internet
start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 27)

#figure out the tickers for each stock

BAC = web.DataReader('BAC', 'google', start=start, end=end)

#Finding ticker names from google searchs

#CitiGroup
C = web.DataReader('C', 'google', start, end)

#Goldman Sachs
GS = web.DataReader('GS', 'google', start, end)

#JPMorgan Chase
JPM = web.DataReader('JPM', 'google', start, end)

#Morgan Stanley
MS = web.DataReader('MS', 'google', start, end)

#Wells Fargo
WFC = web.DataReader('WFC', 'google', start, end)

#Create a list of tickers in alphabetical order#
tickers = ['BAC','C','GS','JPM','MS','WFC']

#use pd.concat to concatenate the bank dataframes into a single data frame
#called bank_stocks.
bank_stocks = pd.concat([BAC,C,GS,JPM,MS,WFC], axis=1, keys=tickers)
print(bank_stocks.head())

bank_stocks.columns.names = ['BankTicker','Stock Info']

border = '-' * 24
print(border + 'Exploratory Data Analysis' + border)
#Question: What is the max Close price of each bank's stock during the time period

max_closing_price = bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
print(max_closing_price)

#Creating a new DataFrame called returns, which will hold all returns
#for each bank's stock.
returns = pd.DataFrame()

#use pandas pct_change() method on close column to create a new column
#that represents the return value#

for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
print(returns.head())

#Creating a pairplot using seaborn of the returns df. What stocks stands out
#to you? Can you figure out why?#

import seaborn as sns
import matplotlib.pyplot as plt
plt.show(sns.pairplot(returns[1:]))

#Back to the main question of which plot stood out and why
#The stocks look relatively normal except the CitiGroup stock that looks
#like a straight line, that represents the citigroup crash #

#using returns we figure out what dates each bank had the best and the worst
#single day returns.

print(returns.idxmin())
#gives series of index/items of min values

print(returns.idxmax())

#look at std of returns, which stock would you classify as the riskiest
#over the entire time period? which was the riskiest for the year 2015?

print(returns.std())

#riskiest is citigroup

#for 2015
returns.ix['2016-01-30':'2016-12-01']

sns.distplot(returns.ix['2016-01-01':'2016-12-31']['MS Return'], color='green', bins=50)
sns.distplot(returns.ix['2016-01-01':'2016-12-31']['C Return'], color='red', bins=50)

