__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
import sourceCode.stock_LSTM
import sourceCode.data_collector as Collector
import math
import numpy as np
import pandas as pd

class DataFormatter():
    """A class for loading and transforming 1D arrays of stock data into normalized windows"""

    def __init__(self, stock, unixStart, unixEnd, seq_len, interval='1m', metric='close'):
        stockData = Collector.getStock(stock, unixStart, unixEnd, interval, metric)
        self.data = stockData
        self.len_data = len(stockData)
        self.len_windows = None
        self.data_range = None
        self.median = None
        self.seq_len = seq_len

    def get_input_windows(self, normalise=True):
        '''
        Create input windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_data() method.
        '''
        if self.data_range == None: self.normalize()
        data_x = []
        for i in range(self.len_data - self.seq_len):
            data_x.append(self.data[i:i+self.seq_len])
        return np.array(data_x)

    def get_output_windows(self):
        if self.data_range == None: self.normalize()
        data_y = []
        for i in range(self.len_data - self.seq_len):
            data_y.append(self.data[self.seq_len+i])
        return np.array(data_y)

    def generate_data(self, seq_len, batch_size, normalise=True):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def normalize(self):
        '''
        Normalise window with a base value of zero
        '''
        self.data_range = max(self.data) - min(self.data)
        self.median = np.median(self.data)
        normalized_data = [(point - self.median) / self.data_range for point in self.data]
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

def predict(stock, unixStart, unixEnd, interval):
    '''
    Predict the stock,  beginning at unixStart and ending at unixEnd
    '''
    # Load model that has performed the best thus far
    model = stock_LSTM()
    model.load_model("saved_models\\25step-4epoch-16batch.h5")

def main():
    start = Collector.convert_to_unix(2019,8,1)
    end = Collector.convert_to_unix(2019,8,31)
    df = DataFormatter('AAPL', start, end, 24, '60m','close')
    data_in = df.get_input_windows()
    data_out = df.get_output_windows()
    print("Input windows:", data_in)
    print("Output windows:", data_out)
    print("Input-output match testing:")
    count = 1
    for x, y in zip(data_in[1:], data_out[:-1]):
        print(count,": ",x[-1]==y)
        count += 1
    
if __name__ == '__main__':
    main()
