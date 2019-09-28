from yahoo_fin.stock_info import get_data, get_quote_table, get_live_price
import pandas as pd
import requests
import arrow
import os
import datetime
import time
import numpy as np

def convert_to_unix(year, month, day, hour=0, minute=0, second=0):
    return int(time.mktime(time.strptime('{}-{}-{} {}:{}:{}'.format(year, month, day, hour, minute, second), '%Y-%m-%d %H:%M:%S')))

def convert_from_unix(unixCode):
    return time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(unixCode))

def getStock(strTicker, unixStart, unixEnd, interval='1m', metric='close'):
    '''
    Use Yahoo API to get stock data from startDate/startTime to endDate/endTime
    Use selected interval if appropriate, smallest useable if not
    Metrics include close, open, high, low
    Returns a 1D array containing data for the selected stock and metric
    '''
    # Get data in .json format
    res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(strTicker,unixStart,unixEnd,interval))
    # Convert .json into nested dictionary format
    data = res.json()
    # Save a sub-dictionary of result so we type less code - everything we care about can be accessed inside body
    body = data['chart']['result'][0]
    #Get the date
    dt = datetime.datetime
    # make a Series (array) of time stamps
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
    # index the info by the time frame
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)

    return np.asarray(df.loc[:][metric])

def getIndustryIndicators(startTime, endTime):
    '''
    return a list of the 11 indicator stocks for each industry
    Stocks must have continuous data within the start and end times.
    '''
    industries = ["basic_materials","communication_services","consumer_cyclical","consumer_defensive","energy","financial_services","healthcare","industrials","real_estate","technology","utilities"]
    pass

def getHighestMarketCap(strIndustry, startTime, endTime):
    '''
    return the stock with the highest market cap in the desired industry
    Make sure that the indcator is available for the given time frame
    yahoo_fin.get_quote_table returns a dictionary that contains market cap data
    '''
    pass

def get_quote_data(symbol, data_range='1d', data_interval='15m'):
    '''
    The allowed ranges are: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
    The allowed intervals are: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo
    '''
    res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval={}".format(symbol,data_range,data_interval))
    # Try replacing 'range=' with 'preiod1=' and 'period2=' and using specific UNIX date/time codes
    # You can get UNIX time code through the datetime module
    data = res.json()
    body = data['chart']['result'][0]
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])

    return df.loc[:, ('open', 'high', 'low', 'close', 'volume')]

if __name__ == "__main__":
    print(get_quote_data('AAPL','1d','1m'))
