from yahoo_fin.stock_info import get_data, get_quote_table, get_live_price
import pandas as pd
import requests
import arrow
import os
import datetime


def getStock(strTicker, startDate, endDate, startTime, endTime, interval):
    ## Need code that takes in parameters above and uses get_quote_data to fulfill requests
    pass

def get_quote_data(symbol='SBIN.NS', data_range='1d', data_interval='15m'):
    '''
    The allowed ranges are: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
    The allowed intervals are: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo
    '''
    res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval={}".format(symbol,data_range,data_interval))
    data = res.json()
    body = data['chart']['result'][0]
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])

    return df.loc[:, ('open', 'high', 'low', 'close', 'volume')]

if __name__ == "__main__":
    print(get_quote_data('AAPL','1d','1m'))