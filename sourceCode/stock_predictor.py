__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
from datetime import time, date, datetime, timedelta
import math
import matplotlib.pyplot as plt
import sourceCode.stock_LSTM as stockModel
import sourceCode.data_collector as DC
import math
import numpy as np
import pandas as pd
import random
from threading import Thread



class DataFormatter():
    """     A class for loading and transforming 1D arrays of stock data into normalized windows    """

    def __init__(self, data, seq_len):
        '''
        data - 1D array of data (input)
        seq_len - the length of each window to be returned
        '''
        self.data = DC.patch_data(data)
        self.data_len = len(data)
        # Range of the data
        self.data_range = None
        # Median of the data
        self.median = None
        self.seq_len = seq_len
        # Go ahead and normalize
        self.data = self.normalize()

    def get_input_windows(self, normalize=True):
        '''
        Create input windows based on object values
        returns an array of sub-arrays which are of length self.seq_len
        '''
        if not normalize: self.de_normalize()
        data_x = []
        if self.data_len < self.seq_len: raise ValueError("Total data length less than sequence length")
        for i in range(self.data_len - self.seq_len):
            data_x.append(self.data[i:i+self.seq_len])
        return np.array(data_x)

    def get_output(self, normalize=True):
        if not normalize: array = self.de_normalize()
        else: array = self.data
        data_y = []
        for i in range(self.data_len - self.seq_len):
            data_y.append(array[self.seq_len+i])
        return np.array(data_y)

    def get_input_generator(self, normalize=True):
        if not normalize: self.de_normalize()
        if self.data_len < self.seq_len: raise ValueError("Total data length less than sequence length")
        for i in range(self.data_len - self.seq_len):
            yield self.data[i:i+self.seq_len]

    def normalize(self, data=[]):
        '''
        Normalise window with a base value of zero
        '''
        if len(data) == 0 : data = self.data
        self.data_range = max(data) - min(data)
        self.median = np.median(data)
        normalized_data = []
        for point in data:
            normalized_data.append ((point - self.median) / self.data_range)
        return np.array(normalized_data)



    def de_normalize(self, data=[]):
        '''
        De-Normalize an array based on the values used to normalize the data the DF was initiated withn
        '''
        if len(data) == 0: data = self.data
        normalized_data = []
        if self.data_range == None: return
        for point in data:
            normalized_data.append (point * self.data_range + self.median)
        return np.array(normalized_data)

    def test_in_out_match(self):
        '''
        test method to make sure inputs and outputs are synced up
        '''
        _in = self.get_input_windows()
        _out = self.get_output()
        for i in range(self.data_len - self.seq_len-1):
            print("Out: "+str(_out[i])+"; In: "+str(_in[i+1][self.seq_len-1])+"; Match: "+str(_out[i]==_in[i+1][self.seq_len-1]))


def plot_results(predicted_data, true_data):
    '''
    Use matplotlib to graph true data and predictions
    Good for debugging, won't be needed for the actual product
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(predicted_data, label='Prediction')
    ax.plot(true_data, label='True Data')
    
    plt.legend()
    plt.show()


def load_model(self, filepath):
	self.model = load_model(filepath)


def predict(input, modelType="mainModel.h5"):
    '''
    Predict the stock
    '''
    # Load model that has performed the best thus far
    model = stockModel.stock_LSTM()
    model.load_model("saved_models\\"+modelType)
    prediction = model.predict_point_by_point(input)
    return prediction


def compileData(stockCollector, seq_len):
    '''
    Get all of the data needed to make a prediction compiled into a 3D numpy array just like the neural net will need it
    Had to use generators in order to get one window from each stock at a time
    Assumptions: stockCollector is fully initialized
    '''
    compiled = []
    window = []
    # Get generator from stock being predicted
    if stockCollector.mainData == []: raise ValueError("")
    collected = stockCollector.mainData
    df = DataFormatter(collected, seq_len)
    mainDataGen = df.get_input_generator()
    # Get generators for competitors
    compGens = []
    for competitor in stockCollector.competitorData: 
        df = DataFormatter(competitor, seq_len)
        compGens.append(df.get_input_generator())
    indicatorGens = []
    # Get generators for indicators
    for indicator in stockCollector.indicatorData: 
        df = DataFormatter(indicator, seq_len)
        indicatorGens.append(df.get_input_generator())
    #Get the window for the current timestep from each generator and make into a 2D array
    for i in range(len(collected)-seq_len):
        window = []
        nex = next(mainDataGen)
        window.append(nex)
        for ind in indicatorGens:
            nex = next(ind)
            window.append(nex)
        for comp in compGens:
            nex = next(comp)
            window.append(nex)
        if np.shape(window) != (16,24): raise ValueError("Missing required data.")
        # Which is then added to a 3D array, which is the returned result
        compiled.append(np.transpose(np.asarray(window)))
    return np.asarray(compiled)

def historical_prediction(stock, startArray, endArray, interval, metric):
    '''
    stock       -   ticker of the stock you want to predict
    startArray  -   start date / time array in the form [year, month, day, hour, minute]
    endArray    -   start date / time array in the form [year, month, day, hour, minute]
    interval    -   Smallest interval of predicted points desired - '1m', '1h', or '1d'
    metric      -   Measure of stock value - either 'open', 'close', 'high', or 'low'
    Note: start and end are inclusive, i.e. they will not "round" to the next date / time
    Note: interval is self-regulating and will throw errors if too small
    '''
	start = convert_to_unix(startArray[0], startArray[1], startArray[2], startArray[3], startArray[4])
	end = convert_to_unix(endArray[0], endArray[1], endArray[2], endArray[3], endArray[4])
	
    dataEnd = datetime.fromtimestamp(end)

    # NeededPoints = time elapsed / interval + 24 (24 points before first prediction) - 1 (don't need to collect end prediction point)
    neededPoints = int((datetime.fromtimestamp(end) - datetime.fromtimestamp(start))/DC.DataCollector.INTERVALS[interval] + 24 - 1)
    try:
        testCollector = DC.DataCollector.fromEndpoint(stock, end, neededPoints, interval, metric)
        df = DataFormatter(testCollector.mainData, 24)
        input = compileData(testCollector, 24)
        output = df.get_output(normalize=False)
    except: print("stock",stock,"does not have enough associated data available to make a prediction for the time selected"); return 
    model = stockModel.stock_LSTM()
    model.load_model("saved_models\\mainModel.h5")
    prediction = predict(input)
	return df.de_normalize(prediction)
    
def future_prediction(stock, end, interval, metric, __dev_start=None, __dev_output=None):
    '''
    stock       -   ticker of the stock you want to predict
    end         -   UNIX-formatted timestamp for ending time/date of prediction
    interval    -   Smallest interval of predicted points desired - '1m', '1h', or '1d'
    metric      -   Measure of stock value - either 'open', 'close', 'high', or 'low'
    Note: prediction will always start at the next interval that does not currently have a prediction
    Note: interval is self-regulating and will throw errors if too small
    '''
    now = datetime.now()
    if __dev_start != None: now = datetime.fromtimestamp(__dev_start)
    if interval == '1m': 
        print('using 1m interval')
        formatted_now = datetime(year=now.year, month = now.month, day = now.day, hour = now.hour, minute=now.minute)
        shortInterval = '1m'
        longInterval = '15m'
        # Furthest prediction is 30 mins, need 24 points per prediction
        shortPredictions = 30;
        shortPoints = shortPredictions*24;
        # Furthest prediction is 360 mins, need 24 points per prediction, divided by 15min interval
        longPredictions = int(360/15)
        longPoints = longPredictions*24;
        shortLongConversion = 15;
    elif interval == '1h': 
        formatted_now = datetime(year=now.year, month = now.month, day = now.day, hour = now.hour)
        print('using 1h interval')
        shortInterval = '1h'
        longInterval = '1d'
        # Furthest prediction is 24 hrs, need 24 points per prediction
        shortPredictions = 24;
        shortPoints = shortPredictions*24;
        # Furthest prediction is 720 hrs, need 24 points per prediction, divided by 24 hr interval
        longPredictions = int(720/24)
        longPoints = longPredictions*24;
        shortLongConversion = 24;
    else : 
        formatted_now = datetime(year=now.year, month = now.month, day = now.day)
        print('using 1d interval')
        shortInterval = '1d'
        longInterval = '1wk'
        # Furthest prediction is 30 days, need 24 points per prediction
        shortPredictions = 30;
        shortPoints = shortPredictions*24;
        # Furthest prediction is 180 days, need 24 points per prediction, divided by 7 day interval
        longPredictions = int(math.round(180/7))
        longPoints = longPredictions*24;
        shortLongConversion = 7;


    collectionEnd = (formatted_now - DC.DataCollector.INTERVALS[shortInterval]).timestamp()
    shortCollector = DC.DataCollector.fromEndpoint(stock, collectionEnd, shortPoints, shortInterval, metric)
    collectionEnd = (formatted_now - DC.DataCollector.INTERVALS[longInterval]).timestamp()
    longCollector = DC.DataCollector.fromEndpoint(stock, collectionEnd, longPoints, longInterval, metric)
    print("Short:",shortCollector.mainData)
    print("Long:",longCollector.mainData)

    shortInputWindows = []
    longInputWindows = []
    shortData = []
    longData = []
    shortData.append(DataFormatter(shortCollector.mainData, 24).data)
    longData.append(DataFormatter(longCollector.mainData, 24).data)
    for i in range(4):
        shortData.append(DataFormatter(shortCollector.competitorData[i], 24).data)
        longData.append(DataFormatter(shortCollector.competitorData[i], 24).data)
    for i in range(11):
        shortData.append(DataFormatter(shortCollector.indicatorData[i], 24).data)
        longData.append(DataFormatter(longCollector.indicatorData[i], 24).data)

    for i in range(1,shortPredictions+1):
        shortWindow = []
        for j in range(16):
            shortStrip = []
            for k in range(1,25):
                shortStrip.append(shortData[j][len(shortData[j])-(k*i)])
            shortWindow.append(shortStrip)
        shortInputWindows.append(shortWindow)

    print(shortInputWindows)
    print(np.shape(shortInputWindows))

    for i in range(1,longPredictions+1):
        longWindow = []
        for j in range(16):
            longStrip = []
            for k in range(1,25):
                longStrip.append(longData[j][len(longData[j])-(k*i)])
            longWindow.append(longStrip)
        longInputWindows.append(longWindow)

    print(longInputWindows)
    print(np.shape(longInputWindows))



    shortPredictionPoints = []
    for i in shortInputWindows:
        shortPredictionPoints.append(predict(np.reshape(np.transpose(np.asarray(i)), (1,24,16,))))

    longPredictionPoints = []
    for i in longInputWindows:
        longPredictionPoints.append(predict(np.reshape(np.transpose(np.asarray(i)), (1,24,16,))))


    plot_results(shortPredictionPoints,[] if __dev_start == None else __dev_output)






    



def main():

    pastStart = DC.convert_to_unix(2019,10,7,12,30)
    pastEnd = DC.convert_to_unix(2019,10,7,13,0)

    end = DC.convert_to_unix(2019,10,28,12,30)

    DC.DataCollector.setup()

    output = DataFormatter(DC.DataCollector.fromDate('AAPL',pastStart, pastEnd, '1m', 'close', False).dateCollect(),24).data


    future_prediction('AAPL',end,'1m','close', pastStart, output)
    
if __name__ == '__main__':
    main()
