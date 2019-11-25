__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"


import os
import json
from datetime import time, date, datetime, timedelta
from keras.backend import clear_session
import math
import matplotlib.pyplot as plt
import sourceCode.stock_LSTM as stockModel
import sourceCode.data_collector as DC
import numpy as np
import pandas as pd
import random
from threading import Thread

environment = os.path.join( os.path.dirname ( __file__), os.path.pardir)



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
        self.data_range = max(data) - min(data)
        # Median of the data
        self.median = np.median(data)
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


def predict(input):
    '''
    Predict the stock
    '''
    # Load model that has performed the best thus far
    model = stockModel.stock_LSTM()
    model.load_model(os.path.join(environment,"data\\mainModel.h5"))
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

def historical_prediction(stock, start, end, interval, metric):
    '''
    stock       -   ticker of the stock you want to predict
    startArray  -   start datetime array of the form [year, month, day, hour, minute]
    endArray    -   end datetime array of the form [year, month, day, hour, minute]
    interval    -   Smallest interval of predicted points desired - '1m', '1h', or '1d'
    metric      -   Measure of stock value - either 'open', 'close', 'high', or 'low'
    Note: start and end are inclusive, i.e. they will not "round" to the next date / time
    Note: interval is self-regulating and will throw errors if too small
    '''
    startDT = datetime.fromtimestamp(start)
    endDT = datetime.fromtimestamp(end)
    numDays = math.ceil((endDT.date() - startDT.date())  / timedelta(days=1))+1
    tempCollector = DC.DataCollector(stock, interval, metric)
    neededPoints = tempCollector.getTimeOnMarket(startDT,endDT)
    neededPoints += 24
    dataEnd = datetime.fromtimestamp(end)

    # NeededPoints = time elapsed / interval + 24 (24 points before first prediction) - 1 (don't need to collect end prediction point)
    print(neededPoints,"points needed for",stock,"prediction from",datetime.fromtimestamp(start),"to",dataEnd)
   
    try:
        testCollector = DC.DataCollector.fromEndpoint(stock, end, neededPoints, interval, metric)
        df = DataFormatter(testCollector.mainData, 24)
        inputx = compileData(testCollector, 24)
    except: return f'stock {stock} does not have enough associated data' 

    prediction = predict(inputx)
    clear_session()
    return df.de_normalize(prediction)
    
def future_prediction(stock, end_unix, interval, metric, start_unix=None):
    '''
    stock       -   ticker of the stock you want to predict
    end         -   UNIX-formatted timestamp for ending time/date of prediction
    interval    -   Smallest interval of predicted points desired - '1m', '1h', or '1d'
    metric      -   Measure of stock value - either 'open', 'close', 'high', or 'low'
    Note: prediction will always start at the next interval that does not currently have a prediction
    Note: interval does not regulate and must be checked before making predictions
    '''
    
    # Set start point of prediction
    if start_unix != None: 
        predictionStart = datetime.fromtimestamp(start_unix)
    else: predictionStart = datetime.now()

    predictionEnd = datetime.fromtimestamp(end_unix)

    neededPoints = DC.DataCollector(stock, interval, metric).getTimeOnMarket(predictionStart,predictionEnd)
## Determine intervals, define long/short variable values
    if interval == '1m': 
        # Reconstruct the date without any time scale smaller than minutes
        formatted_start = datetime(year=predictionStart.year, month=predictionStart.month, day=predictionStart.day, hour=predictionStart.hour, minute=predictionStart.minute)
        formatted_end   = datetime(year = predictionEnd.year, month = predictionEnd.month, day = predictionEnd.day, hour = predictionEnd.hour, minute = predictionEnd.minute)
        
        # Furthest short prediction is 30 mins, need 24 points per prediction
        shortInterval = '1m'
        max_short_output = 30
        short_output_count = max_short_output if neededPoints > max_short_output else neededPoints
        short_input_count = short_output_count*24

        # Furthest prediction is 360 mins, need 24 points per prediction, divided by 15min interval
        longInterval = '15m'
        max_long_output = int(360/15) 
        intervalConversion = DC.DataCollector.INTERVALS[longInterval] // DC.DataCollector.INTERVALS[shortInterval]
        long_output_count = max_long_output if math.ceil(neededPoints//intervalConversion) > max_long_output else neededPoints//intervalConversion
        long_input_count = long_output_count*24

    elif interval == '1h': 
        # Reconstruct the date without any time scale smaller than hours
        formatted_start = datetime(year=predictionStart.year, month=predictionStart.month, day=predictionStart.day, hour=predictionStart.hour)
        formatted_end   = datetime(year = predictionEnd.year, month = predictionEnd.month, day = predictionEnd.day, hour = predictionEnd.hour)

        # Furthest prediction is 24 hrs, need 24 points per prediction
        shortInterval = '1h'
        max_short_output = 24;
        short_output_count = max_short_output if neededPoints > max_short_output else neededPoints
        short_input_count = short_output_count*24;

        # Furthest prediction is 720 hrs, need 24 points per prediction, divided by 24 hr interval
        longInterval = '1d'
        max_long_output = int(720/24)
        intervalConversion = DC.DataCollector.INTERVALS[longInterval] // DC.DataCollector.INTERVALS[shortInterval]
        long_output_count = max_long_output if math.ceil(neededPoints/intervalConversion) > max_long_output else neededPoints//intervalConversion
        long_input_count = long_output_count*24;

   

    # Set the endpoint for collection to be one interval before the current time
    shortCollectionEnd = (formatted_start - DC.DataCollector.INTERVALS[shortInterval]).timestamp()

    # Setup a DataCollector for each interval
    shortCollector = DC.DataCollector.fromEndpoint(stock, shortCollectionEnd, short_input_count, shortInterval, metric)
    

## Compile the data into windows
    shortInputWindows = []

    shortData = []

    # Normalize data for short and long prediction streams and
    # Add each stock's data array to the overall data array for the central stock

    ## Set up the model
    model=stockModel.stock_LSTM()
    model.load_model(os.path.join(environment,'data\\mainModel.h5'))

    if short_output_count > 0:
        shortData.append(DataFormatter(shortCollector.mainData, 24).data)
        for i in range(4):
            shortData.append(DataFormatter(shortCollector.competitorData[i], 24).data)
        for i in range(11):
            shortData.append(DataFormatter(shortCollector.indicatorData[i], 24).data)

        # Get the interleaved data points and format them into windows
        for i in range(1,short_output_count+1):
            shortWindow = []
            for j in range(16):
                shortStrip = []
                for k in range(1,25):
                    shortStrip.append(shortData[j][len(shortData[j])-(k*i)])
                shortWindow.append(shortStrip)
            shortInputWindows.append(shortWindow)

        ## Create and launch short prediction threads
        num_threads = 8
        short_predictions_per_thread = math.ceil(short_output_count / num_threads)

        # Create an empty predictions list (to be filled by prediction threads)
        shortPredictionsList = np.asarray(np.zeros((short_output_count,)))

        # Set up thread array and indexing info
        threads = []
        start = 0
        end = short_predictions_per_thread-1
        for i in range(num_threads):
            # Create and start new thread 
            threads.append(Thread(target=predictions, args=(model, shortInputWindows, range(start,end), shortPredictionsList, 1)))
            threads[-1].start()

            # Set new start and end points
            start = end
            end += short_predictions_per_thread
        
            # Correct end point if it goes past the number of outputs we're supposed to have
            if end >= short_output_count:
                end = short_output_count-1

    ## Wait for threads to finish
        for i in range(num_threads):  threads[i].join()

    if long_output_count > 0:
        longCollectionEnd =  (formatted_start - DC.DataCollector.INTERVALS[ longInterval]).timestamp()
        longCollector  = DC.DataCollector.fromEndpoint(stock,  longCollectionEnd,  long_input_count,  longInterval, metric)
        longInputWindows = []
        longData = []

        longData.append(DataFormatter(longCollector.mainData, 24).data)
        for i in range(4):
            longData.append(DataFormatter(longCollector.competitorData[i], 24).data)
        for i in range(11):
            longData.append(DataFormatter(longCollector.indicatorData[i], 24).data)

        for i in range(1,long_output_count+1):
            longWindow = []
            for j in range(16):
                longStrip = []
                for k in range(1,25):
                    longStrip.append(longData[j][len(longData[j])-(k*i)])
                longWindow.append(longStrip)
            longInputWindows.append(longWindow)

        ## Create and laumch long prediction threads
        # Create an empty predictions list (to be filled by prediction threads)
        longPredictionsList = np.asarray(np.zeros(long_output_count*intervalConversion,))

        # Set up thread array and indexing info
        long_predictions_per_thread = math.ceil(long_output_count / num_threads)
        threads=[]
        start=0
        end = long_predictions_per_thread-1

        for i in range(num_threads):
            #Create and start new thread
            threads.append(Thread(target=predictions, args=(model, longInputWindows, range(start,end), longPredictionsList, 15)))
            threads[-1].start()

            # Set new start and end points
            start = end
            end += long_predictions_per_thread

             # Correct end point if it goes past the number of outputs we're supposed to have
            if end >= long_output_count:
                end = long_output_count-1

        # Wait for threads to finish
        for i in range(num_threads): threads[i].join()

        # Patch missing points between long interval predictions
        segmentStart = 0
        segmentEnd = intervalConversion-1
        for i in range(long_output_count):
            # But only if there's no data in the range between long predictions
            longPredictionsList = DC.average_fill(longPredictionsList, segmentStart,segmentEnd)
            segmentStart = segmentEnd
            segmentEnd += intervalConversion
            if segmentEnd > long_input_count: 
                segmentEnd = long_input_count

    if long_output_count > 0: predictionsList = longPredictionsList
    else: predictionsList = np.asarray(np.zeros((short_output_count*intervalConversion)))
    # Replace values in long predictions list with values in short prediction list (if they are the same point in time)
    for i in range(len(shortPredictionsList)):
        predictionsList[i] = shortPredictionsList[i]

    #Clean up prediction model to prepare for next prediction
    clear_session()

    return predictionsList

def make_prediction (stock, startArray, endArray, interval, metric):
    startDT = datetime(startArray[0], startArray[1], startArray[2],startArray[3],startArray[4])
    endDT = datetime(endArray[0], endArray[1], endArray[2],endArray[3],endArray[4])
    now = datetime(datetime.now().year,datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute)
    if endDT > now: 
        tempCollector = DC.DataCollector(stock, interval, metric)
        required_intervals = tempCollector.getTimeOnMarket(now,endDT) / 2
        required_historic_start = tempCollector.findTradeStart(now, required_intervals)
        historic_start = (startDT+DC.DataCollector.INTERVALS[interval]).timestamp() if startDT < required_historic_start else (required_historic_start+DC.DataCollector.INTERVALS[interval]).timestamp()

        h_predictions = historical_prediction(stock, historic_start, now.timestamp(), interval, metric)
        f_predictions = future_prediction(stock, endDT.timestamp(), interval, metric)
        df = DataFormatter(h_predictions,24)

        output = DC.DataCollector.fromDate(stock, historic_start, now.timestamp(), interval, metric, False).dateCollect()
        return [output, np.asarray(np.append(h_predictions,df.de_normalize(f_predictions),0))]

    else: 
        output = DC.DataCollector.fromDate(stock, startDT.timestamp(), endDT.timestamp(), interval, metric, False).dateCollect()
        return [output, historical_prediction(stock, startDT.timestamp(), endDT.timestamp(), interval, metric)]



def predictions(model, windows, indices, predictions, interval=1):
    for i in indices:
         predictions[i*interval-1] = model.model.predict_on_batch(np.reshape(np.transpose(np.asarray(windows[i])), (1,24,16,)))


def get_actual_data(stock_ticker, start, end, interval, metric):
    output = DC.DataCollector.fromDateArray(stock_ticker,start,end,interval,metric,False).dateCollect()
    return output


def main():
    '''
    Main method
    '''
    start = DC.convert_to_unix(2019,11,12,12,30)
    end =   DC.convert_to_unix(2019,11,25,16,0)

    DC.DataCollector.setup()
    output, prediction = make_prediction('DOW',[2019,11,21,12,30],[2019,11,21,16,0],'1m','close')
    plot_results(prediction, output)
    #print(stockModel.performance(output_df.normalize(h_prediction),output_df.normalize(output)))
    '''
    start = [2019,11,11,9,30]
    end = [2019,11,12,16,0]

    output = DC.DataCollector.fromDateArray('AAPL',start,end,'1m','close',False).dateCollect()
    prediction = historical_prediction('AAPL',start,end,'1m','close')
    plot_results(prediction, output)
    print('Output ' + str(len(output)))
    print('Prediction ' + str(len(prediction)))
    print(stockModel.performance(prediction, output))
    '''

    # DC.DataCollector.fromEndpoint('AAPL',pastEnd, 90, '15m', 'close')

    
if __name__ == '__main__':
    main()




