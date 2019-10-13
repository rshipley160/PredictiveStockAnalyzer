#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import requests
import arrow
import os
import datetime
import time
import numpy as np
import threading

# HTML Parse imports
from lxml import html  
import json
from collections import OrderedDict

# Market cap scrape imports
import urllib.request
import urllib.parse
import urllib.error
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl
import ast
import os


def convert_to_unix(year, month, day, hour=0, minute=0):
    '''
    Return an integer representing the unix timestamp equivalent of the specified date & time
    '''
    return int(time.mktime(time.strptime('{}-{}-{} {}:{}:0'.format(year, month, day, hour, minute), '%Y-%m-%d %H:%M:%S')))


def convert_from_unix(unixCode):
    '''
    Return a formatted string displaying the date/time represented by the given unix timestamp
    '''
    return time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(unixCode))


def patch_data(data):
    '''
    Replace NaN values in a 1D array by averaging surrounding datapoints
    Does not mutate the original array
    Throws a ValueError if the data is over 40% NaN or has NaN endpoints
    Returns the patched array of data
    '''
    # Determine if this function can be used on the data
    percent_nan = np.count_nonzero(np.isnan(data)) / len(data)
    if percent_nan > 0.4: raise ValueError("This data contains too many NaN values to patch.")
    if percent_nan == 0: return data
    if np.isnan(data[0]) or np.isnan(data[-1]): raise ValueError("Data array must have numeric end points")

    # Copy the data so we don't mutate it
    patched = data.copy()
    start = None

    # Find holes in the data and patch using average_fill
    for i in range(1,len(patched)):
        # At each nan encountered after a valid number...
        if np.isnan(data[i]) and start == None:
            # Set the start of the sub-array as the last valid number
            # having start set to a number indicates we need an end point
            start = i-1
        # If we have a start and run into another valid number...
        elif not np.isnan(data[i]) and start != None:
            # Set the end of the sub-array
            end = i
            # fill the gaps between the start and end points
            patched = average_fill(patched, start, end)
            # Reset start so that we don't keep looking for an end point
            start = None

    # Give back the patched data
    return patched


def average_fill(data, start, end):
    '''
    Returns the given data array with values in the range (start, end) (both exclusive) replaced with average deviation between end points
    Does not mutate original array
    '''
    # Calculate the average
    dev = (data[end] - data[start]) / (end-start)

    dataCopy = data.copy()
    # Overwrite selected indices bast on preceding value and deviation
    for i in range(start+1, end):
        dataCopy[i] = dataCopy[i-1]+dev
    return dataCopy


class DataCollector:
    '''     Static variables used in static and instance methods    '''

    # Will contain a dictionary of the form {industry:stocks}...
    # where industries is a string in INDUSTRIES
    # and stocks is a list of stocks in that industry
    industryStocks = {}

    # Will contain a dictionary of the form {industry:stocks} after init...
    # Where industry is a string in INDUSTRIES
    # and stocks is a dictionary of the form {ticker:marketCap}
    marketCapDict = {}

    # Will contain a dictionary of the form {industry:sortedStocks} after init...
    # where industries is a string in INDUSTRIES
    # and sortedStocks is a list of tuples containing of the form (ticker, marketCap) sorted by marketCap 
    sortedStocks = {}

    # Will contain a list of industry indicator stock tickers after init
    industryIndicators = []

    # A constant list of all the financial sectors Yahoo divides stocks into
    INDUSTRIES = ['basic_materials','communication_services','consumer_cyclical','consumer_defensive','energy','financial_services','healthcare','industrials','real_estate','technology','utilities']
    
    # Boolean that indicates whether the static initialization has been completed
    initialized = False

    '''     Static methods      '''
    @staticmethod
    def setup():
        DataCollector.loadIndustryStocks()

        DataCollector.loadIndustryIndicators()

        DataCollector.initialized = True


    @staticmethod
    def loadIndustryStocks():
        '''
        Load industryStocks dictionary with stocks labeled by industry
        '''
        industryFile = open('data\Industries.txt','r')
        industries = {}
        for industryLine in industryFile.readlines():
            industry = industryLine.rstrip().split(',')
            DataCollector.industryStocks[industry[0]] = industry[1:]


    @staticmethod
    def getHighestMarketCap(industry):
        '''
        return the stock ticker that has the highest market cap in the desired industry
        Make sure that the indcator is available
        '''
        tickers = DataCollector.industryStocks[industry]
        if industry not in DataCollector.marketCapDict.keys():
            DataCollector.__loadMarketCapDict(industry)
        # sortedStocks is a list of tuples containing key value pairs that are ordered by market cap
        if industry not in DataCollector.sortedStocks.keys():
            DataCollector.sortedStocks[industry] = sorted(DataCollector.marketCapDict[industry].items(), key=lambda x:x[1], reverse=True)

        # Go down the sorted list of stocks 
        # and pick the first one that has good data available
        for stock in DataCollector.sortedStocks[industry]:
            # Use two weeks ending at the current hour as the threshold
            today = datetime.datetime.now().strftime("%Y/%m/%d/%H").split("/")
            end = convert_to_unix(today[0],today[1],today[2],today[3])
            backdate = (datetime.datetime.now() - datetime.timedelta(weeks=2)).strftime("%Y/%m/%d/%H").split("/")
            start = convert_to_unix(backdate[0],backdate[1],backdate[2],backdate[3])


            collector = DataCollector(stock[0], start, end, '1h', 'close')
            try:
                # Run these two lines to make sure data can be collected
                raw = collector.collect()
                # and patched if needed
                patched = patch_data(raw)
                return stock[0]
            except ValueError:
                continue


    @staticmethod
    def __loadMarketCapDict(industry):
        '''
        Helper method to getHighestMarketCap
        Splits sector loading among multiple threads
        Load marketCapDict[industry] with stock values
        Uses multiple threads to load stocks in parallel
        '''
        if industry not in DataCollector.industryStocks:
            DataCollector.loadIndustryStocks()

        numStocks = len(DataCollector.industryStocks[industry])
        

        threads = []
        numThreads = 8
        section_length = int( numStocks / numThreads)

        start = 0
        end = section_length
        for i in range(numThreads):
            threads.append(threading.Thread(target=DataCollector.__loadCapSection, args=(start,end,industry)))
            threads[i].start()
            start = end
            end = end + section_length
            if end > numStocks: end = numStocks 
        for i in range(numThreads):
            threads[i].join()


    @staticmethod
    def __loadCapSection(start, end, industry):
        '''
        Helper method to __loadMarketCapDict
        This is what each thread uses to load a segment of the stocks in the industry
        '''
        stocks = DataCollector.industryStocks[industry]
        for i in range(start,end):
            try:
                if industry not in DataCollector.marketCapDict.keys():
                    DataCollector.marketCapDict[industry] = {}
                DataCollector.marketCapDict[industry][stocks[i]] = DataCollector.convertCap(DataCollector.marketCap(stocks[i]))
            except ValueError: pass
            except urllib.error.HTTPError: pass


    @staticmethod
    def loadIndustryIndicators():
        '''
        loads industryIndicators list with the ticker of the stock in each industrythat has the highest market cap and has data available
        Note: Takes a while to load, time dependent on how many threads can be executed concurrently
        '''
        startTime = time.time()
        for industry in DataCollector.INDUSTRIES:
            print("industry: "+industry)
            stock = DataCollector.getHighestMarketCap(industry)
            DataCollector.industryIndicators.append(stock)
        print(time.time() - startTime)


    @staticmethod
    def convertCap(strMarketCap):
        '''
        Returns the float conversion of the given market cap string
        '''
        if type(strMarketCap) == float: return strMarketCap
        if type(strMarketCap) != str: raise ValueError
        
        unit = strMarketCap[-1:]
        if unit in ('K','M','B','T'):
            amount = float(strMarketCap[:-1])
            if unit == 'K':
                amount *= 1e6
            if unit == 'M':
                amount *= 1e6
            if unit == 'B':
                amount *= 1e9
            if unit == 'T':
                amount *= 1e12
        else: amount = float(strMarketCap)
        return amount

    
    @staticmethod
    def marketCap(ticker):
        '''
        Return the market cap string scraped from yahoo finance
        Note: Not our original source code, however it is unattributed on the website, as well as unlicensed
        Source: https://www.promptcloud.com/blog/how-to-scrape-yahoo-finance-data-using-python/ 
        '''
        #Create stock page url
        url = "https://in.finance.yahoo.com/quote/{}?".format(ticker)
        # Making the website believe that you are accessing it using a Mozilla browser
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        # Creating a BeautifulSoup object of the HTML page for easy extraction of data.

        soup = BeautifulSoup(webpage, 'html.parser')
        html = soup.prettify('utf-8')
        company_json = {}
        other_details = {}
        marketCap = None
        for td in soup.findAll('td', attrs={'data-test': 'MARKET_CAP-value'}):
            for span in td.findAll('span', recursive=False):
                marketCap = span.text.strip()
                print('found '+marketCap)
        return marketCap

    '''     Instance Methods        '''
    def __init__(self, ticker, unixStart, unixEnd, interval, metric):
        self.start = unixStart
        self.end = unixEnd
        self.interval = interval
        self.metric = metric
        self.stock = ticker
        
    
    def collect(self, patch=True):
        '''
        Use Yahoo API to get stock data
        Metrics include close, open, high, low
        Returns a 1D array containing data for the selected stock and metric
        '''
        # Get data in .json format
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(self.stock,self.start,self.end,self.interval)
        res = requests.get(url)
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
        try: 
            patched = patch_data(np.asarray(df.loc[:][self.metric]))
        except: patched = np.asarray(df.loc[:][self.metric])
        return patched


def main():
    pass


def parse(sector):

    ctx = ssl.create_default_context()
    ctx.check_hostname = false
    ctx.verify_mode = ssl.cert_none

    url = "http://finance.yahoo.com/ms_{}".format(sector)
    response = requests.get(url, verify=False)
    print ("Parsing %s"%(url))
    time.sleep(4)
    parser = html.fromstring(response.text)
    print(parser)
    '''
	summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
	summary_data = OrderedDict()
	other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
	summary_json_response = requests.get(other_details_json_link)
	try:
		json_loaded_summary =  json.loads(summary_json_response.text)
		y_Target_Est = json_loaded_summary["quoteSummary"]["result"][0]["financialData"]["targetMeanPrice"]['raw']
		earnings_list = json_loaded_summary["quoteSummary"]["result"][0]["calendarEvents"]['earnings']
		eps = json_loaded_summary["quoteSummary"]["result"][0]["defaultKeyStatistics"]["trailingEps"]['raw']
		datelist = []
		for i in earnings_list['earningsDate']:
			datelist.append(i['fmt'])
		earnings_date = ' to '.join(datelist)
		for table_data in summary_table:
			raw_table_key = table_data.xpath('.//td[contains(@class,"C(black)")]//text()')
			raw_table_value = table_data.xpath('.//td[contains(@class,"Ta(end)")]//text()')
			table_key = ''.join(raw_table_key).strip()
			table_value = ''.join(raw_table_value).strip()
			summary_data.update({table_key:table_value})
		summary_data.update({'1y Target Est':y_Target_Est,'EPS (TTM)':eps,'Earnings Date':earnings_date,'ticker':ticker,'url':url})
		return summary_data
	except:
		print ("Failed to parse json response")
		return {"error":"Failed to parse json response
    '''


if __name__ == "__main__":
   main()
