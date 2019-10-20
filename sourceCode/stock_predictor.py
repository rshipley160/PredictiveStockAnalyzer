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

class DataFormatter():
    """A class for loading and transforming 1D arrays of stock data into normalized windows"""

    def __init__(self, data, seq_len, ):
        self.data = data
        self.len_data = len(data)
        self.len_windows = None
        self.data_range = None
        self.median = None
        self.seq_len = seq_len

    def get_input_windows(self, normalise=True):
        '''
        Create input windows
        '''
        if self.data_range == None and normalise: self.normalize()
        data_x = []
        if self.len_data < self.seq_len: raise ValueError("Total data length less than sequence length")
        for i in range(self.len_data - self.seq_len):
            data_x.append(self.data[i:i+self.seq_len])
        return np.array(data_x)

    def get_output_windows(self):
        if self.data_range == None: self.normalize()
        data_y = []
        for i in range(self.len_data - self.seq_len):
            data_y.append(self.data[self.seq_len+i])
        return np.array(data_y)

    def get_input_generator(self, normalize=True):
        if self.data_range == None and normalize: self.normalize()
        if self.len_data < self.seq_len: raise ValueError("Total data length less than sequence length")
        for i in range(self.len_data - self.seq_len):
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


def plot_results(predicted_data, true_data):
    '''
    Use matplotlib to graph true data and predictions
    Good for debugging, won't be needed for the actual product
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def load_model(self, filepath):
	print('[Model] Loading model from file %s' % filepath)
	self.model = load_model(filepath)


def predict(input):
    '''
    Predict the stock
    '''
    # Load model that has performed the best thus far
    model = stockModel.stock_LSTM()
    model.load_model("saved_models\\25step-4epoch-16batch.h5")
    prediction = model.predict_point_by_point(input)
    return prediction


def compileData(stockCollector, seq_len):
    compiled = []
    window = []
    collected = stockCollector.collect()
    df = DataFormatter(collected, seq_len)
    mainDataGen = df.get_input_generator()
    compGens = []
    for competitor in stockCollector.competitorData: 
        df = DataFormatter(competitor, seq_len)
        print(len(list(df.get_input_generator())))
        compGens.append(df.get_input_generator())
    indicatorGens = []
    for indicator in stockCollector.indicatorData: 
        df = DataFormatter(indicator, seq_len)
        print(len(list(df.get_input_generator())))
        indicatorGens.append(df.get_input_generator())
    for i in range(len(collected)-seq_len):
        window = []
        window.append(next(mainDataGen))
        for ind in indicatorGens:
            window.append(next(ind))
        for comp in compGens:
            window.append(next(comp))
        compiled.append(np.transpose(np.asarray(window)))
    return np.asarray(compiled)


def main():
    start = DC.convert_to_unix(2019,1,1)
    end = DC.convert_to_unix(2019,1,31)
    DC.DataCollector.setup()
    # Put your test code here

    # Code for prediction of 5 random stocks from each industry
    '''
    for industry in DC.DataCollector.INDUSTRIES:
        for i in range(5):
            index = random.randint(0,len(DC.DataCollector.industryStocks[industry])-1)
            stock = DC.DataCollector.industryStocks[industry][index]
            mainStock = DC.DataCollector(stock, start, end, '1h', 'close')
            input = compileData(mainStock, 24)
            output = DataFormatter(mainStock.mainData, 24).get_output_windows()
            predictions = predict(input)
            plot_results(predictions,output)
            print(stockModel.performance(predictions,output))
    '''
    
if __name__ == '__main__':
    main()
