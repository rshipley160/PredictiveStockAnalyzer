# Predictive Stock Analyzer
## Purpose
This is a collaborative project assigned to Sequoia Kanies, Zac Byers, Brandon Do, and Riley Shipley as their senior capstone project. The purpose of this project is to give us as students a chance to put all that we have learned over the course of our studies to use in a useful and interesting project.
Our assignment is to create a __Predictive Stock Analyzer__ that is capable of predicting future stock values and is easy for the average person to understand and use to view stocks that they are interested in
## Implementation
The way we plan to achieve these goals is through a system of interconnected components that all come together to make a system that is easy to use, efficient, unique, and hopefully accurate, though we have been assured that the predictions do not have to be accurate in order to get a good grade on the project as a whole. The three main components of our system are described in the following sections
### Data collection
The data collection component is an original Python program that uses the [yahoo fin](http://theautomatic.net/yahoo_fin-documentation/) module to retrieve historical and live market data. It will be used by the other two components to collect stock data as needed.
### Stock prediction
The prediction component is also written in Python, largely because it is a language we are all comfortable with that has fairly extensive machine learning support. The neural network is heavily based on [this implementation of an LSTM prediction network](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction) by @jaungiers, but it has been adapted to use data from multiple stocks to try to predict a single stock's future value. The approach we've taken is to use historical data from the stock being predicted and its competitors inside and outside the industry in order to produce a prediction. Our hope is that using this new combination of factors. 
### User Interface
We have made our user interface using C# and the WinForms system in Visual Studio in an effort to make production more efficient than other methods and platforms. The goal of the user interface is to remain simple and intuitive while requiring minimal user effort to generate and view desired stock predictions.
