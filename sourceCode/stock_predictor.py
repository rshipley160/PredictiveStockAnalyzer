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
        print("Normalized:",self.data)

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
    print("Collected main data:",collected)
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

def historical_prediction(stock, start, end, interval, metric):
    '''
    stock       -   ticker of the stock you want to predict
    start       -   UNIX-formatted timestamp for starting time/date of prediction
    end         -   UNIX-formatted timestamp for ending time/date of prediction
    interval    -   Smallest interval of predicted points desired - '1m', '1h', or '1d'
    metric      -   Measure of stock value - either 'open', 'close', 'high', or 'low'
    Note: start and end are inclusive, i.e. they will not "round" to the next date / time
    Note: interval is self-regulating and will throw errors if too small
    '''
    dataEnd = datetime.fromtimestamp(end)

    # NeededPoints = time elapsed / interval + 24 (24 points before first prediction) - 1 (don't need to collect end prediction point)
    neededPoints = int((datetime.fromtimestamp(end) - datetime.fromtimestamp(start))/DC.DataCollector.INTERVALS[interval] + 24 - 1)
    try:
        testCollector = DC.DataCollector.fromEndpoint(stock, end, neededPoints, interval, metric)
    except: print("stock",stock,"does not have enough data available to make a prediction"); return 
    df = DataFormatter(testCollector.mainData, 24)
    input = compileData(testCollector, 24)
    output = df.get_output(normalize=False)
    model = stockModel.stock_LSTM()
    model.load_model("saved_models\\mainModel.h5")
    prediction = predict(input)
    plot_results(prediction,df.normalize(output))
    print(stockModel.performance(df.de_normalize(prediction),output))
    

    



def main():

    start = DC.convert_to_unix(2019,10,7,12,30)
    end = DC.convert_to_unix(2019,10,7,14,30)

    DC.DataCollector.setup()

    for industry in DC.DataCollector.INDUSTRIES:
        stock = DC.DataCollector.industryStocks[industry][random.randint(0,len(DC.DataCollector.industryStocks[industry]))]
        historical_prediction(stock,start,end,'1m','high')
    
if __name__ == '__main__':
    main()
