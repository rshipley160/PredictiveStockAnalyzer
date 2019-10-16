import os
import math
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from numpy import newaxis
from LSTM_base.core.utils import Timer
from LSTM_base.core.data_processor import DataLoader
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

        print('[Model] Model Compiled')

        return self.model

    def train(self, x, y, epochs=1, batch_size=32, save_dir="saved_models", save_name=None):
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

        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)

    def load_model(self, filepath):
	    print('[Model] Loading model from file %s' % filepath)
	    self.model = load_model(filepath)
        
    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
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
    pass
