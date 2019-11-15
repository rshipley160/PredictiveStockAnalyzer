import os
import math
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model as load
from keras.callbacks import EarlyStopping, ModelCheckpoint
import stock_components.sourceCode.stock_predictor as Predictor
from stock_components.sourceCode.data_collector import convert_to_unix, DataCollector as DC
import random

environment = os.path.join( os.path.dirname ( __file__), os.path.pardir)

debugging = False

def debug(string):
    if debugging == True: print(str(string))

class stock_LSTM:
    """ This class is intended to include strictly the methods used for creating and interacting with the model itself"""
    ''' All of the parameters passed to these methods must be normalized and shaped correctly'''

    def __init__(self):
        self.model = Sequential()

    def build_model(self, neurons=100, input_steps=49, dropout_rate = 0.2, loss="mse", optimizer="adam"):
        self.model.add(LSTM(neurons, input_shape=(input_steps, 16), return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(neurons, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(neurons, return_sequences=False))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])

        debug('[Model] Model Compiled')

        return self.model

    def train(self, x, y, epochs=1, batch_size=32, save_dir=os.path.join(environment,"saved_models"), save_name=None, save=True):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
        if save_name == None: save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        else: save_fname = save_name
        callbacks = [
	        EarlyStopping(monitor='val_accuracy', patience=2),
	        ModelCheckpoint(filepath=save_fname, monitor='val_accuracy', save_best_only=True)
        ]
        self.model.fit(
	        x,
	        y,
	        epochs=epochs,
	        batch_size=batch_size,
	        callbacks=callbacks,
        )

        debug('[Model] Training Completed.') 

        if save:
            self.model.save(save_fname)
            debug('Model saved as %s' % save_fname)

    def load_model(self, filepath):
	    debug('[Model] Loading model from file %s' % filepath)
	    self.model = load(filepath)
        
    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        debug('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def performance(predicted_data, true_data):
    if len(predicted_data) != len(true_data): 
        print("Prediction data list and true data list differ in length")
        return 0
    distances = []
    for pred, true in zip(predicted_data, true_data):
        distances.append(abs(true-pred)/(true+1))
    if len(distances) == 0: return 0
    return 1 - sum(distances) / len(distances)

if __name__ == "__main__":
    
    debugging = True
    model = stock_LSTM()
    DC.setup()
    
    #Model prediction
    start = convert_to_unix(2009,10,10)
    end = convert_to_unix(2019,10,12)
    model.load_model(os.path.join(environment,'data\\mainModel.h5'))
    for industry in DC.INDUSTRIES:
        for stock in range(len(DC.industryStocks[industry])):
            try:
                stock = DC.industryStocks[industry][stock]
                collector = DC(stock, start, end, '1mo', 'close')
            except:
                continue
            input = Predictor.compileData(collector, 24)
            output = Predictor.DataFormatter(collector.mainData, 24).get_output_windows()
            predictions = model.predict_point_by_point(input)
            print(stock+": "+str(performance(predictions,output)))
            plot_results(predictions, output)

    '''   
    # Model training 
    start = convert_to_unix(2019,3,31)
    end = convert_to_unix(2019,9,28)
    model.build_model(input_steps=24)
    for industry in DC.INDUSTRIES:
        try:
            stock = DC.industryStocks[industry][random.randint(0,len(DC.industryStocks[industry])-1)]
            collector = DC(stock, start, end, '1h', 'close')
        except:
            continue
        input = Predictor.compileData(collector, 24)
        output = Predictor.DataFormatter(collector.mainData, 24).get_output_windows()
        model.train(input, output, 16, 16, save=False)
        model.model.save('saved_models\\25step-16epoch-16batch-hourly-1each.h5')
    '''

