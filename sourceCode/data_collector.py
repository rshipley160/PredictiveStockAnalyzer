#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import requests
import arrow
import os
import numpy as np
import threading

# HTML Parse imports
from lxml import html  
import json
from collections import OrderedDict
from datetime import time, date, datetime, timedelta
import time as t

# Market cap scrape imports
import urllib.request
import urllib.parse
import urllib.error
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl
import ast
import os.path
from os import path
import pickle
import math
import sys

printDebug = False

def debug(string):
    '''
    Used instead of print statements so I can turn all of the prints off with a single change
    '''
    global printDebug
    if printDebug: print(string)


def convert_to_unix(year, month, day, hour=0, minute=0):
    '''
    Return an integer representing the unix timestamp equivalent of the specified date & time
    '''
    return int(t.mktime(t.strptime('{}-{}-{} {}:{}:0'.format(year, month, day, hour, minute), '%Y-%m-%d %H:%M:%S')))


def convert_from_unix(unixCode):
    '''
    Return a formatted string displaying the date/time represented by the given unix timestamp
    '''
    return t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(unixCode))


def patch_data(data):
    '''
    Replace NaN values in a 1D array by averaging surrounding datapoints
    Does not mutate the original array
    Throws a ValueError if the data is over 40% NaN or has NaN endpoints
    Returns the patched array of data
    '''
    # Determine if this function can be used on the data
    if len(data) < 3: return data 
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
    '''     
    Class used to collect all data required to make predictions about a stock
    See constructor for detailed explanation of use
    '''

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

    # A constant list of all the financial sectors Yahoo divides stocks into
    INDUSTRIES = ['basic_materials','communication_services','consumer_cyclical','consumer_defensive','energy','financial_services','healthcare','industrials','real_estate','technology','utilities']

    # Constant dictionary to convert string intervals into timedelta objects
    INTERVALS = {'1m':timedelta(minutes=1),'2m':timedelta(minutes=2),'5m':timedelta(minutes=5),'15m':timedelta(minutes=15),'30m':timedelta(minutes=30),'1h':timedelta(hours=1),'1d':timedelta(days=1),'1wk':timedelta(days=7),'1mo':timedelta(days=30)}

    # Boolean that indicates whether the static initialization has been completed
    initialized = False

    '''     Static methods      '''
    @staticmethod
    def setup(force_reload=False):
        try:
            if DataCollector.initialized: return
            debug("starting setup")
            DataCollector.loadIndustryStocks()

            if (force_reload == False):
                DataCollector.loadMarketCap()
            else:
                DataCollector.reloadMarketCap()

            DataCollector.initialized = True
            debug("setup complete")

            return True

        except:
            return False

    @staticmethod
    def loadIndustryStocks():
        '''
        Load industryStocks dictionary with stocks labeled by industry
        '''
        debug("Loading industryStocks dictionary...")
        environment = os.path.join( os.path.dirname ( __file__), os.path.pardir)

        industryFile = open(os.path.join(environment,'data\Industries.txt'),'r')
        industries = {}
        for industryLine in industryFile.readlines():
            industry = industryLine.rstrip().split(',')
            DataCollector.industryStocks[industry[0]] = industry[1:]
        debug("industryStocks loaded")


    @staticmethod
    def loadMarketCap():
        '''
        load the sorted and unsorted market cap dictionaries
        '''
        debug("Loading market cap dictionaries...")
        if len(DataCollector.industryStocks.keys()) == 0: DataCollector.loadIndustryStocks()
        for industry in DataCollector.industryStocks.keys():
            if industry not in DataCollector.marketCapDict.keys(): DataCollector.__loadMarketCapDict(industry)
                # sortedStocks is a list of tuples containing key value pairs that are ordered by market cap
            if industry not in DataCollector.sortedStocks.keys():
                DataCollector.sortedStocks[industry] = sorted(DataCollector.marketCapDict[industry].items(), key=lambda x:x[1], reverse=True)
        debug("Market cap dictionaries loaded")

    @staticmethod
    def reloadMarketCap():
        '''
        load the sorted and unsorted market cap dictionaries
        '''
        debug("Reloading market cap dictionaries...")
        if len(DataCollector.industryStocks.keys()) == 0: DataCollector.loadIndustryStocks()
        for industry in DataCollector.industryStocks.keys():
            DataCollector.__loadMarketCapDict(industry, True)
                # sortedStocks is a list of tuples containing key value pairs that are ordered by market cap
            DataCollector.sortedStocks[industry] = sorted(DataCollector.marketCapDict[industry].items(), key=lambda x:x[1], reverse=True)
        debug("Market cap dictionaries loaded")


    @staticmethod
    def __loadMarketCapDict(industry, force_reload=False):
        '''
        Helper method to getHighestMarketCap
        Splits sector loading among multiple threads
        Load marketCapDict[industry] with stock values
        Uses multiple threads to load stocks in parallel
        Loads from dictionary if dictionary is available
        '''
        debug("Loading market cap dict for "+industry+"...")

        script_path = os.path.realpath(__file__)
        environment = os.path.join( os.path.dirname ( __file__), os.path.pardir)

        if path.exists(os.path.join(environment,("data\\marketCapDict_"+industry+".dat"))) and not force_reload:
            debug("File found. Loading dictionary")
            obj_file = open(os.path.join(environment,("data\\marketCapDict_"+industry+".dat")),'rb')
            DataCollector.marketCapDict[industry] = pickle.load(obj_file)
            obj_file.close()
        else: 
            debug("No file found. Creating dictionary")            

            if industry not in DataCollector.industryStocks.keys():
                DataCollector.loadIndustryStocks()
            numStocks = len(DataCollector.industryStocks[industry])
        
            #Split the work between 8 threads
            threads = []
            numThreads = 8
            
            section_length = math.ceil(numStocks / numThreads)
            debug(str(numStocks)+"to be loaded")
            start = 0
            end = section_length
            for i in range(numThreads):
                # Each thread will get some portion of the total list of stock tickers in the industry
                # and loads that section into the dictionary
                threads.append(threading.Thread(target=DataCollector.__loadCapSection, args=(start,end,industry)))
                threads[i].start()
                # Set the start and end of the next thread's section
                start = end
                end = end + section_length
                if end > numStocks: end = numStocks 
            for i in range(numThreads):
                threads[i].join()

            #Save to a binary file that the program knows how to open for reuse
            if not path.exists(os.path.join(environment,"data")): os.makedirs("data")
            obj_file = open(os.path.join(environment,("data\\marketCapDict_"+industry+".dat")),'wb')
            pickle.dump(DataCollector.marketCapDict[industry], obj_file)
            obj_file.close()

        debug("Market cap dict for "+industry+" loaded")


    @staticmethod
    def __loadCapSection(start, end, industry):
        '''
        Helper method to __loadMarketCapDict
        This is what each thread uses to load a segment of the stocks in the industry
        '''
        debug("Loading market cap for stocks "+str(start)+"-"+str(end)+" from "+industry+"...")
        stocks = DataCollector.industryStocks[industry]
        for i in range(start,end):
            try:
                if industry not in DataCollector.marketCapDict.keys():
                    DataCollector.marketCapDict[industry] = {}
                DataCollector.marketCapDict[industry][stocks[i]] = DataCollector.convertCap(DataCollector.marketCap(stocks[i]))
            except ValueError: pass
            except urllib.error.HTTPError: pass
        debug("Market caps "+str(start)+"-"+str(end)+" from "+industry+" loaded")


    @staticmethod
    def convertCap(strMarketCap):
        '''
        Returns the float conversion of the given market cap string
        '''
        debug("Converting market cap...")
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
        debug("Market cap converted")
        return amount

    
    @staticmethod
    def marketCap(ticker):
        '''
        Return the market cap string scraped from yahoo finance
        Note: Not our original source code, however it is unattributed on the website, as well as unlicensed
        Source: https://www.promptcloud.com/blog/how-to-scrape-yahoo-finance-data-using-python/ 
        '''
        debug("Getting market cap for "+ticker+"...")
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
        debug("Market cap retrieved for "+ticker)
        return marketCap
        
    @staticmethod
    def collectStock(stock, dateTimeIndex, interval, metric, patch=True):
        '''
        Use Yahoo API to get stock data
        Metrics include close, open, high, low
        Returns a 1D array containing data for the selected stock and metric
        '''
        debug("collecting data for "+stock+"...")
        start = int(datetime.fromisoformat(dateTimeIndex[0]).timestamp())
        end = int((datetime.fromisoformat(dateTimeIndex[-1])+DataCollector.INTERVALS[interval]).timestamp())
        # Get data in .json format
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(stock,start,end,interval)
        res = requests.get(url)
        # Convert .json into nested dictionary format
        data = res.json()
        # Save a sub-dictionary of result so we type less code - everything we care about can be accessed inside body
        chart = data['chart']
        result = chart['result']
        if result == None: raise ValueError
        body = result[0]
        try:
            #dt = list(map(lambda x: arrow.get(x).to('EDST').datetime.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"), body['timestamp']))
            dt = list(map(lambda x: str(datetime.fromtimestamp(x)), body['timestamp']))
            df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
        except: raise ValueError("unable to retrieve stock values")
        # index the info by the time frame
        raw = []
        for index in dateTimeIndex:
            try: 
                raw.append(df.loc[index][metric])
            except:
               raw.append(np.nan)
        if patch:
            try: 
                patched = patch_data(np.asarray(raw))
            except: patched = np.asarray(raw)
        else: patched = np.asarray(raw)
        debug("collection completed")
        return patched

    '''     Constructors        '''
    @classmethod
    def fromDate(cls, ticker, unixStart, unixEnd, interval, metric, __full_setup = True):
        '''
        Set up the DataCollecter using specified parameters
        ticker - ticker of the stock you wish to collect data from
        unixStart - precise date and time you wish to begin retrieving data, converted to a unix timestamp
        unixEnd - precise date and time you wish to stop retrieving data, converted to a unix timestamp
        interval - the time interval used, or the granularity of the data
                Allowed intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk
        metric - the type of stock metric you'd like to use - open, high, low, close
        __full_setup - whether or not this constructor is a full DataCollector
                Determines if it can attempt setup of the static class
                And if the full attributes of the stock will be loaded on initialization
                Some load methods use shallow DataCollectors, this is to avoid unwanted recursion
        '''
        if not DataCollector.initialized and __full_setup: DataCollector.setup()
        debug("Intializing stock "+ticker+"...")
        obj = cls(ticker, interval, metric)
        obj.start = unixStart
        obj.end = unixEnd
        obj.interval = interval
        obj.metric = metric
        obj.stock = ticker
        obj.competitors = []
        obj.industry = None
        obj.capIndex = None
        obj.competitorNames = []
        obj.competitorData = []
        obj.indicatorNames = []
        obj.indicatorData  = []
        obj.mainData = []
        obj.dateTimeIndex = []
        if __full_setup: 
            obj.mainData = obj.dateCollect()
            obj.industry = obj.getIndustry()
            obj.loadCompetitors()
            obj.loadIndicators()
            obj.capIndex = obj.getMarketCapIndex()
        debug("Intialization complete")
        return obj


    @classmethod
    def fromDateArray(cls, ticker, startArray, endArray, interval, metric, __full_setup = True):
        '''
        Set up the DataCollecter using specified parameters
        ticker - ticker of the stock you wish to collect data from
        startArray - array in the form [year, month, day, hour, minute] that specifies start date/time
        endArray - array in the form [year, month, day, hour, minute] that specifies end date/time
        interval - the time interval used, or the granularity of the data
                Allowed intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk
        metric - the type of stock metric you'd like to use - open, high, low, close
        __full_setup - whether or not this constructor is a full DataCollector
                Determines if it can attempt setup of the static class
                And if the full attributes of the stock will be loaded on initialization
                Some load methods use shallow DataCollectors, this is to avoid unwanted recursion
        '''
        if not DataCollector.initialized and __full_setup: DataCollector.setup()
        debug("Intializing stock "+ticker+"...")
        obj = cls(ticker, interval, metric)
        obj.start = convert_to_unix(startArray[0], startArray[1], startArray[2], startArray[3], startArray[4])
        obj.end = convert_to_unix(endArray[0], endArray[1], endArray[2], endArray[3], endArray[4])
        obj.interval = interval
        obj.metric = metric
        obj.stock = ticker
        obj.competitors = []
        obj.industry = None
        obj.capIndex = None
        obj.competitorNames = []
        obj.competitorData = []
        obj.indicatorNames = []
        obj.indicatorData  = []
        obj.mainData = []
        obj.dateTimeIndex = []
        if __full_setup: 
            obj.mainData = obj.dateCollect()
            obj.industry = obj.getIndustry()
            obj.loadCompetitors()
            obj.loadIndicators()
            obj.capIndex = obj.getMarketCapIndex()
        debug("Intialization complete")
        return obj

    @classmethod
    def fromEndpoint(cls, ticker, endPoint, numPoints, interval, metric, __full_setup=True):
        if not DataCollector.initialized and __full_setup: DataCollector.setup()
        debug("Intializing stock "+ticker+"...")
        obj = cls(ticker, interval, metric)
        obj.competitors = []
        obj.industry = None
        obj.capIndex = None
        obj.competitorNames = []
        obj.competitorData = []
        obj.indicatorNames = []
        obj.indicatorData  = []
        obj.mainData = []
        obj.dateTimeIndex = []
        if __full_setup: 
            obj.collect(endPoint, numPoints)
            obj.industry = obj.getIndustry()
            obj.loadCompetitors()
            obj.loadIndicators()
            obj.capIndex = obj.getMarketCapIndex()
        debug("Intialization complete")
        return obj

    
    def __init__(self, ticker, interval, metric):
        debug("Intializing stock "+ticker+"...")
        self.interval = interval
        self.metric = metric
        self.stock = ticker
        self.mainData = []

    '''         Instance Methods            '''
    def getIndustry(self):
        '''
        Returns the industry this stock belongs to
        '''
        if self.industry != None: return self.industry
        debug("Finding industry for "+self.stock+"...")
        for industry in DataCollector.INDUSTRIES:
            for stock in DataCollector.industryStocks[industry]:
                if stock == self.stock: 
                    debug("Industry Found")
                    return industry 
        

    def getPoint(self, timestamp, _interval='1m'):
        '''
        Return the data point for the timestamp and interval given
        Interval is needed because requests have to contain two separate timestamps to return a single value
        '''
        # Get data in .json format
        if DataCollector.INTERVALS[_interval] >= timedelta(days=1):
            nextDate = date.fromtimestamp(timestamp) + DataCollector.INTERVALS[_interval]
            nextDateTime = int(datetime(nextDate.year, nextDate.month, nextDate.day, 0, 0, 0, 0).timestamp())
        else:
            nextDateTime = int((datetime.fromtimestamp(timestamp) + DataCollector.INTERVALS[_interval]).timestamp())
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(self.stock,timestamp,nextDateTime,_interval)
        res = requests.get(url)
        # Convert .json into nested dictionary format
        data = res.json()
        # Save a sub-dictionary of result so we type less code - everything we care about can be accessed inside body
        try:
            point = data['chart']['result'][0]['indicators']['quote'][0][self.metric][0]
        except: 
            point = np.nan
        return point 


    def getHours(self):
        '''
        Returns the start and end of regular trading time for this stock
        '''
        start = int((datetime.now().replace(hour=6, minute=0,second=0)-timedelta(days=4)).timestamp())
        end = int((datetime.now().replace(hour=20,minute=0,second=0)+timedelta(days=3)).timestamp())
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(self.stock, start, end,'1d')
        res = requests.get(url)
        debug(str(datetime.fromtimestamp(start))+" "+str(datetime.fromtimestamp(end))+" "+url)
        # Convert .json into nested dictionary format
        data = res.json()
        # Save a sub-dictionary of result so we type less code - everything we care about can be accessed inside body 
        chart = data['chart']
        result = chart['result']
        if result == None: raise ValueError
        start = datetime.fromtimestamp(int(result[0]['meta']["currentTradingPeriod"]['regular']['start'])).time()
        end = datetime.fromtimestamp(int(result[0]['meta']["currentTradingPeriod"]['regular']['end'])).time()
        return start, end


    def getTimeOnMarket(self, start, end, interval=None):
        '''
        Returns the number of intervals that the stock was actively traded in the indicated time span
        start - datetime object representing start date and time
        end - datetime object representing end date and time
        interval - optional, allows user to specify interval.
                    uses object interval if not set
        Returns - an integer representing the number of intervals that the stock was traded in the period given
        Note: Converts any interval >= a day into stock days - only as long as trade end - trade start
        '''
        if interval == None: interval = self.interval
        currentDay = start.date()
        market_start, market_end = self.getHours()
        marketTime = timedelta(days=0)

        fixedInterval = DataCollector.INTERVALS[interval]
        if DataCollector.INTERVALS[interval] >= timedelta(days=1):
            fixedInterval = ((datetime.combine(date.today(), market_end) - datetime.combine(date.today(),market_start)) *  (DataCollector.INTERVALS[interval] / timedelta(days=1)))

        time_on_market = 0

        while(currentDay.day <= end.day):
            if currentDay.weekday() > 4:
                currentDay += timedelta(days=1)
                continue
            if start.time() > market_end and currentDay.day == start.day:
                currentDay += timedelta(days=1)
                continue
            if end.time() < market_start and currentDay.day == end.day:
                currentDay += timedelta(days=1)
                continue


            day_start = market_start
            if market_start < start.time() <= market_end and currentDay == start.date(): day_start = start.time()

            day_end = market_end
            if market_start <= end.time() < market_end and currentDay == end.date(): day_end = end.time()

            marketTime = datetime.combine(date.today(), day_end) - datetime.combine(date.today(),day_start)

            #print(currentDay.isoformat(), day_start.isoformat(), day_end.isoformat(), marketTime)
            currentDay += timedelta(days=1)

            time_on_market += math.ceil(marketTime/fixedInterval) 
        
        return time_on_market


    def findTradeStart(self, end, numIntervals, interval=None):
        '''
        Return the start date that makes it so that there are numPoints intervals on the market between the start and end date
        end - datetime object specifying endpoint
        numIntervals - the number of intervals of trading time needed
        interval - interval used. Defaults to stock interval
        '''
        if interval == None: interval = self.interval
        intervalTotal = 0
        currentDay = end.date()
        market_start, market_end = self.getHours()
        while (intervalTotal < numIntervals):
            if currentDay.weekday() > 4:
                currentDay -= timedelta(days=1)
                continue
            if end.time() < market_start and currentDay.day == end.day:
                currentDay -= timedelta(days=1)
                continue

            day_end = market_end
            if market_start <= end.time() < market_end and currentDay == end.date(): day_end = end.time()

            day_intervals = math.floor((datetime.combine(currentDay, day_end) - datetime.combine(currentDay, market_start)) / DataCollector.INTERVALS[interval])
            if day_intervals + intervalTotal >= numIntervals:
                intervals_needed = numIntervals - intervalTotal
                return datetime.combine(currentDay, day_end) - (intervals_needed * DataCollector.INTERVALS[interval])
            else:
                intervalTotal += day_intervals
                currentDay -= timedelta(days=1)


    def collect(self, endPoint, numPoints, _interval=None):
        '''
        Sets the dateTimeIndex of this DataCollector to include a number of datapoints which ends at endPoint
        endPoint  - UNIX timestamp object representing the endpoint of the data to collect
        numPoints - the number of points needed
        '''
        currentPoint = datetime.fromtimestamp(endPoint)
        debug("collecting "+str(numPoints)+" points for "+self.stock+" ending at "+str(currentPoint))
        startTime, endTime = self.getHours()
        if _interval == None: _interval = self.interval
        count = 0
        index = []
        points = []
        currentPoint = datetime.fromtimestamp(endPoint).date()
        end = datetime.fromtimestamp(endPoint).time()
        attempts = 0
        while (count < numPoints):
            # Get start and end of trading day
            startStamp = int(datetime.combine(currentPoint, startTime).timestamp())
            endStamp = int(datetime.combine(currentPoint, end).timestamp()) + 3600
            url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(self.stock,startStamp,endStamp,_interval)
            debug(str(datetime.combine(currentPoint, startTime))+" "+str(datetime.fromtimestamp(endStamp))+" "+url)
            res = requests.get(url)
            # Convert .json into nested dictionary format
            data = res.json()
            try:
                chart = data['chart']
                result = chart['result']
                body = result[0]
                dt = list(map(lambda x: str(datetime.fromtimestamp(x)), body['timestamp']))
                df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
                dt.reverse()
                attempts = 0
                for ind in dt:
                    value = df.loc[ind][self.metric]
                    #print(ind, value)
                    if value == np.nan: continue
                    if datetime.fromisoformat(ind) > datetime.fromtimestamp(endPoint): continue
                    '''
                    if end < datetime.fromisoformat(ind).time():
                        print(datetime.fromisoformat(ind),end)
                        break
                    '''
                    index.append(ind)
                    points.append(value)
                    count += 1
                    if count >= numPoints: break
            except : pass
            currentPoint -= timedelta(days=1)
            end = endTime
           

        index.reverse()
        points.reverse()
        #print(index)
        #print(points)
        self.dateTimeIndex = index
        self.mainData = patch_data(points)


    def dateCollect(self, patch=True, dateTimeIndex = None):
        '''
        Use Yahoo API to get stock data
        Metrics include close, open, high, low
        Returns a 1D array containing data for the selected stock and metric
        '''
        if self.mainData != []: return self.mainData
        debug("collecting data for "+self.stock+"...")
        # Get data in .json format
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}".format(self.stock,int(self.start),int(self.end),self.interval)
        debug(str(datetime.fromtimestamp(self.start))+" "+ str(datetime.fromtimestamp(self.end))+" "+url)
        res = requests.get(url)
        # Convert .json into nested dictionary format
        data = res.json()
        # Save a sub-dictionary of result so we type less code - everything we care about can be accessed inside body
        chart = data['chart']
        result = chart['result']
        if result == None: raise ValueError("TEST TEST TEST")
        body = result[0]
        try:
            dt = list(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"), body['timestamp']))
            df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
        except: raise ValueError("unable to retrieve stock values")
        if dateTimeIndex == None:
            self.dateTimeIndex = dt

            if patch:
                try: 
                    patched = patch_data(np.asarray(df.loc[:][self.metric]))
                except: patched = np.asarray(df.loc[:][self.metric])
            else: patched = np.asarray(df.loc[:][self.metric])
            debug("collection completed")
            return patched
        else:
            # index the info by the time frame
            raw = []
            for time in dateTimeIndex:
                try: 
                    raw.append(df.loc[time][self.metric])
                except:
                    raw.append(np.nan)
            if patch:
                try: 
                    patched = patch_data(np.asarray(raw))
                except: patched = np.asarray(raw)
            else: patched = np.asarray(raw)
            debug("collection completed")
            return patched


    def getMarketCapIndex(self):
        '''
        Return this stock's index in the static sortedStocks list, which is ordered by market caps
        '''
        if self.capIndex != None: return self.capIndex
        for i in range(len(DataCollector.sortedStocks[self.getIndustry()])):
            if DataCollector.sortedStocks[self.getIndustry()][i][0] == self.stock: return i


    def loadIndicators(self):
        '''
        Load the list of indicator names and stock data as specified by this object's attributes
        '''
        debug("Loading industry indicators...")
        if self.indicatorNames != [] and self.indicatorData != []: return
        if self.dateTimeIndex == []: self.mainData = self.collect()
        for industry in DataCollector.INDUSTRIES:
            nextIndustry = False
            # Go down the sorted list of stocks 
            # and pick the first one that has good data available
            for stock in DataCollector.sortedStocks[industry]:
                try:
                    # Run these two lines to make sure data can be collected
                    raw = DataCollector.collectStock(stock[0],self.dateTimeIndex, self.interval, self.metric)
                    # and patched if needed
                    patched = patch_data(raw)
                    debug(stock[0]+" has the highest market cap for "+industry)
                    self.indicatorNames.append(stock[0])
                    self.indicatorData.append(patched)
                    break
                except ValueError:
                    continue
        debug("Industry indicators loaded")


    def loadCompetitors(self):
        '''
        Load a list of 4 stock tickers which are the nearest on both sides of the stock
        Also loads data for each competitor into a list
        returns the 4 nearest in market cap if it is not possible to get two on each side
        Throws a value error if 4 competitor stocks cannot be identified
        '''
        index = self.getMarketCapIndex()
        compsLeft = 4
        movingDown = True
        competitors = []
        seek_index = index
        if index == None: return []
        while (compsLeft > 0):
            if seek_index <= 0:
                seek_index = index
                movingDown = False
            if compsLeft <= 2 and movingDown:
                seek_index = index
                movingDown = False
            if movingDown:
                seek_index -= 1
            else: 
                seek_index += 1
            try:
                stock = DataCollector.sortedStocks[self.getIndustry()][seek_index][0]
                debug("Attempting to collect "+stock+"...")

                raw = DataCollector.collectStock(stock, self.dateTimeIndex, self.interval, self.metric)
                patched = patch_data(raw)
                self.competitorNames.append(stock[0])
                self.competitorData.append(patched)
                compsLeft -= 1
                debug("Collection succeeded")
            except ValueError: debug("Collection failed")
            if seek_index >= len(DataCollector.sortedStocks[self.getIndustry()]): raise ValueError


def parse(sector):
    '''
    May be useful in the future for getting the list of stocks in each sector
    Right now it is not used at all
    '''
    ctx = ssl.create_default_context()
    ctx.check_hostname = false
    ctx.verify_mode = ssl.cert_none

    url = "http://finance.yahoo.com/ms_{}".format(sector)
    response = requests.get(url, verify=False)
    debug ("Parsing %s"%(url))
    time.sleep(4)
    parser = html.fromstring(response.text)
    debug(parser)
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
		debug ("Failed to parse json response")
		return {"error":"Failed to parse json response
    '''

def main():
    stock = DataCollector('ORCL','1d','close')
    start = datetime(2019,11,25,21,30)
    end = datetime(2019,11,29,10,30)
    print(stock.getTimeOnMarket(start,end))



if __name__ == "__main__":
   printDebug = True
   main()
