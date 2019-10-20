__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
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
        self.data = data
        self.data_len = len(data)
        # Range of the data
        self.data_range = None
        # Median of the data
        self.median = None
        self.seq_len = seq_len
        # Go ahead and normalize
        self.normalize()

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
        if not normalize: self.de_normalize()
        data_y = []
        for i in range(self.data_len - self.seq_len):
            data_y.append(self.data[self.seq_len+i])
        return np.array(data_y)

    def get_input_generator(self, normalize=True):
        if not normalize: self.de_normalize()
        if self.data_len < self.seq_len: raise ValueError("Total data length less than sequence length")
        for i in range(self.data_len - self.seq_len):
            yield self.data[i:i+self.seq_len]

    def normalize(self):
        '''
        Normalise window with a base value of zero
        '''
        self.data_range = max(self.data) - min(self.data)
        self.median = np.median(self.data)
        normalized_data = []
        for point in self.data:
            normalized_data.append ((point - self.median) / self.data_range)
        self.data = np.array(normalized_data)

    def de_normalize(self):
        '''
        Converts the stock data from normalized form (zero-centered, [-1, 1] range) to its original form
        Must have been normalized first
        '''
        if self.data_range == None: return
        self.data = [point * self.data_range + self.median for point in self.data]

    def de_normalize(self, newData):
        '''
        De-Normalize an array based on the values used to normalize the data the DF was initiated withn
        '''
        normalized_data = []
        for point in newData:
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
    '''
    compiled = []
    window = []
    # Get generator from stock being predicted
    collected = stockCollector.collect()
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
        window.append(next(mainDataGen))
        for ind in indicatorGens:
            window.append(next(ind))
        for comp in compGens:
            window.append(next(comp))
        # Which is then added to a 3D array, which is the returned result
        compiled.append(np.transpose(np.asarray(window)))
    return np.asarray(compiled)

def main():
    start = DC.convert_to_unix(2019,9,29)
    end = DC.convert_to_unix(2019,10,12)
    DC.DataCollector.setup()
    # Put your test code here

    for stock in ('AAPL','INTC','MSFT'):
        mainStock = DC.DataCollector(stock, start, end, '1h', 'close')
        df = DataFormatter(mainStock.mainData, 24)
        df.test_in_out_match()

        #input = compileData(mainStock, 24)
        #_out = df.get_output()
        #_pred = predict(input)
        #predictions = df.de_normalize(_pred)
        #output = df.de_normalize(_out)
        #outSeries = []
        #predSeries = []
        #for i in range(24):
        #    predSeries.append(np.nan)
        #outSeries = mainStock.mainData
        #predSeries = np.append(predSeries, predictions)
        #plot_results(predSeries,outSeries)

        #print(stock+"- "+str(stockModel.performance(_pred,_out)))
    
if __name__ == '__main__':
    main()
