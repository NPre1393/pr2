import sys
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from tensorflow.keras import backend as K

import tensorflow as tf
import tensorflow_probability as tfp

import networkx as nx
import math
import itertools
from itertools import chain

import collections
import pickle

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU

from sklearn import linear_model, svm, model_selection
import glob
import os

QmemodelurrmsegraphPath = "/content/drive/My Drive/models/qmemodelurrmsegraph.pkl"
MsemodelurrmsegraphPath = "/content/drive/My Drive/models/msemodelurrmsegraph.pkl"

MultivarQmemodelrrmsePath = "/content/drive/My Drive/models/multivarqmemodelrrmse.pkl"

MultivarMsemodelrrmsePath = "/content/drive/My Drive/models/multivarmsemodelrrmse.pkl"
MultivarGrumodelrrmsePath = "/content/drive/My Drive/models/multivargrumodelrrmse.pkl"
MultivarLstmmodelrrmsePath = "/content/drive/My Drive/models/multivarlstmmodelrrmse.pkl"

MsemodelurrmsePath = "/content/drive/My Drive/models/msemodelurrmse.pkl"

MsemodelmultiurrmsePath = "/content/drive/My Drive/models/msemodelmultiurrmse.pkl"
MsemodelmultiurfstatPath = "/content/drive/My Drive/models/msemodelmultiurfstat.pkl"

MultivarMsemodelmultimvrmsePath = "/content/drive/My Drive/models/multivarmsemodelmultimvrmse.pkl"
# MultivarGrumodelmultimvrmsePath = "/content/drive/My Drive/models/multivargrumodelmultimvrmse.pkl"
MultivarLstmmodelmultimvrmsePath = "/content/drive/My Drive/models/multivarlstmmodelmultimvrmse.pkl"

MultivarLinearmodelmultimvrmsePath = "/content/drive/My Drive/models/multivarlinearmodelmultimvrmse.pkl"
MultivarLinearmodelmultirrmsePath = "/content/drive/My Drive/models/multivarlinearmodelmultirrmse.pkl"
MultivarSvmmodelmultimvrmsePath = "/content/drive/My Drive/models/multivarsvmmodelmultimvrmse.pkl"
MultivarSvmmodelmultirrmsePath = "/content/drive/My Drive/models/multivarsvmmodelmultirrmse.pkl"

GrurnnmodelurrmsegraphPath = "/content/drive/My Drive/models/grurnnmodelurrmsegraph.pkl"
GrurnnmodelroutputPath = "/content/drive/My Drive/models/grurnnmodelrrmse.pkl"
GrurnnmodeluroutputPath = "/content/drive/My Drive/models/grurnnmodelurrmse.pkl"

MultivarGrurnnmodelmultimvrmsePath = "/content/drive/My Drive/models/multivargrurnnmodelmultimvrmse.pkl"

df = pd.read_csv("/content/drive/My Drive/data/pricesvolumes.csv")
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]

df.drop(df.columns[cols],axis=1,inplace=True)
df.fillna(0, inplace=True)
allcols = df.columns.tolist()
df[allcols[1:]] = df[allcols[1:]].apply(pd.to_numeric).apply(lambda x: x/x.mean(), axis=0)

print(df.columns.tolist())
# sys.exit()

inputbatchsize = 5000
p = 200
q = 200

percentilenum = 10
numepochs = 50
# numepochs = 1


def restricted_mse_model(lag):
    model = Sequential()
    model.add(Dense(units=2*lag, activation='relu', kernel_initializer='normal', bias_initializer='zeros', input_dim=Ypast.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(lag/2, activation='linear', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(lag/2, activation='relu', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='normal', bias_initializer='zeros'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # model.compile(loss='mse', optimizer='adam', metrics=['msd'])
    return model

def quadratic_mean_error(y_true, y_pred):
    sumofsquares = 0
    currpercentile = 0
    prevpercentile = 0
    for i in range(10, 110, percentilenum):
        prevpercentile = currpercentile
        currpercentile = tfp.stats.percentile(y_true, q=i)
        booleaninterpercentile = tf.logical_and(tf.less(y_true,currpercentile),tf.greater(y_true,prevpercentile))
        trueslice = tf.boolean_mask(y_true, booleaninterpercentile)
        predslice = tf.boolean_mask(y_pred, booleaninterpercentile)
        #sumofsquares += tf.to_float(K.square(K.mean(K.square(predslice - trueslice), axis=-1)))
        sumofsquares += tf.cast((K.square(K.mean(K.square(predslice - trueslice), axis=-1))), dtype=tf.float32)
    return K.sqrt(sumofsquares/10)

def restricted_qme_model(lag):
    model = Sequential()
    model.add(Dense(units=2*lag, activation='relu', kernel_initializer='normal', bias_initializer='zeros', input_dim=Ypast.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(lag/2, activation='linear', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(lag/2, activation='relu', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='normal', bias_initializer='zeros'))

    model.compile(optimizer='adam', loss=quadratic_mean_error, metrics=['mse'])
    return model

def getstockdata(dfall, dfout, lag=200):
    Ypast = []
    Ycurr = []

    for i in range(-inputbatchsize, 0):
        y = dfout.iloc[i]
        x = []
        for dfone in dfall:
            x = x + dfall[dfone].iloc[i - lag:i].tolist()
        Ypast.append(x)
        Ycurr.append(y)

    Ypast = np.vstack(Ypast)
    Ycurr = np.vstack(Ycurr)

    Ycurr = Ycurr.reshape(Ycurr.shape[0], )

    return Ypast,Ycurr

def fit_rnn(X, y, n_lag, n_batch, nb_epoch, n_neurons, networktype):
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()

    if(networktype=="GRU"):
        model.add(GRU(2 * n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
        model.add(Dropout(0.5))
        # model.add(LSTM(n_neurons))
        model.add(GRU(n_neurons, stateful=True, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GRU(n_neurons, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(n_neurons))
        model.add(Dropout(0.5))
        model.add(Dense(1))
    if (networktype == "LSTM"):
        model.add(LSTM(2 * n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(n_neurons))
        model.add(Dropout(0.5))
        model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
    return model

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		# forecast = np.array(forecasts[i])
		# forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecasts[i])
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

def preprocess(datacol,scaler):
    raw_values = datacol.values.flatten()

    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    scaled_values = scaler.fit_transform(diff_values).flatten()
    # scaled_values = scaled_values.reshape(len(scaled_values), 1)
    return pd.Series(scaled_values)


def series_to_supervised(df, n_in=1):
    n_vars = 1 if type(df) is list else df.shape[1]
    cols = []
    names = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    Ypast = concat(cols, axis=1)
    Ypast.columns = names
    Ypast.dropna(inplace=True)
    return Ypast


def forecast_gru(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	forecast = model.predict(X, batch_size=n_batch)
	return forecast[0,0]

# evaluate the persistence model
def make_forecasts(model, n_batch, test):
	forecasts = list()
	for i in range(len(test)):
		forecast = forecast_gru(model, test[i,:], n_batch)
		forecasts.append(forecast)
	return forecasts


msemodelmultiurrmse = collections.defaultdict(dict)
msemodelmultiurfstat = collections.defaultdict(dict)
# Multi-cause DNN outputs
msemodelmultiurrmse['AEM_prices'] = {('ORCL_prices', 'T_prices', 'WWD_prices'): 0.666794415502078, ('ABT_prices', 'MSFT_prices', 'WWD_prices'): 0.62846103713634, ('MSFT_prices', 'ORCL_prices', 'T_prices'): 0.6161435744009811, ('ABT_prices', 'ORCL_prices', 'T_prices'): 0.6149284749354357, ('ABT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.6259400834531001, ('ABT_prices', 'T_prices', 'WWD_prices'): 0.5796425051963208, ('ABT_prices', 'MSFT_prices', 'ORCL_prices'): 0.6539380864919132, ('AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.47402648760439237, ('MSFT_prices', 'T_prices'): 0.64835422972639, ('AFG_prices', 'MSFT_prices'): 0.6779950337571623, ('AFG_prices', 'ORCL_prices', 'WWD_prices'): 0.5994242407586627, ('MSFT_prices', 'WWD_prices'): 0.6069621186414063, ('MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.6099013672943597, ('MSFT_prices', 'ORCL_prices'): 0.5196472202704134, ('AFG_prices', 'ORCL_prices'): 0.8522725206575281, ('ABT_prices', 'MSFT_prices'): 0.7247662101240245, ('ABT_prices', 'ORCL_prices'): 0.702838941539504, ('ABT_prices', 'WWD_prices'): 0.5423148543959331, ('MSFT_prices', 'T_prices', 'WWD_prices'): 0.620164980984247, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.5506162781662303, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.6278103945179586, ('ORCL_prices', 'T_prices'): 0.40375898099359253, ('T_prices', 'WWD_prices'): 0.7076292891007344, ('ABT_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.6211659958018467, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.6128981613030056, ('ORCL_prices', 'WWD_prices'): 0.640523062030687, ('ABT_prices', 'ORCL_prices', 'WWD_prices'): 0.5697741204143657}
msemodelmultiurfstat['AEM_prices'] = {('ORCL_prices', 'T_prices', 'WWD_prices'): 0.061240575279726824, ('ABT_prices', 'MSFT_prices', 'WWD_prices'): 0.15323968758112808, ('MSFT_prices', 'ORCL_prices', 'T_prices'): 0.05227784020424836, ('ABT_prices', 'ORCL_prices', 'T_prices'): 0.14296047457112546, ('ABT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.06526875835079654, ('ABT_prices', 'T_prices', 'WWD_prices'): 0.2208029652019143, ('ABT_prices', 'MSFT_prices', 'ORCL_prices'): 0.07477902886788887, ('AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.09624089340782725, ('MSFT_prices', 'T_prices'): 0.6408072775647656, ('AFG_prices', 'MSFT_prices'): 0.38401870796284093, ('AFG_prices', 'ORCL_prices', 'WWD_prices'): 0.06856382921719585, ('MSFT_prices', 'WWD_prices'): 0.14665982205273576, ('MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.09328237524717609, ('MSFT_prices', 'ORCL_prices'): 0.5499946884592325, ('AFG_prices', 'ORCL_prices'): 0.10100676471636978, ('ABT_prices', 'MSFT_prices'): 0.20860185663305156, ('ABT_prices', 'ORCL_prices'): 0.14599573769700444, ('ABT_prices', 'WWD_prices'): 0.2833487213425767, ('MSFT_prices', 'T_prices', 'WWD_prices'): 0.14103393580475182, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.14019584856592918, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.1271385362200985, ('ORCL_prices', 'T_prices'): 0.9948792948447821, ('T_prices', 'WWD_prices'): 0.5033639152032303, ('ABT_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.052758990208022986, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.10621156427645742, ('ORCL_prices', 'WWD_prices'): 0.08657926031203939, ('ABT_prices', 'ORCL_prices', 'WWD_prices'): 0.23353960167297083}
msemodelmultiurrmse['CAT_prices'] = {('ABT_prices', 'MSFT_prices', 'WWD_prices'): 0.5509734761805747, ('ABT_prices', 'T_prices', 'UTX_prices'): 0.5723565177405446, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.47492259572254863, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices'): 0.5314029962005545, ('AAPL_prices', 'ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.42071619863900517, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.420622625155708, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.4308965386836419, ('AAPL_prices', 'ABT_prices', 'MCD_prices', 'T_prices', 'UTX_prices', 'WWD_prices'): 0.38054472668156303, ('AAPL_prices', 'MCD_prices'): 0.4071148085915043, ('ABT_prices', 'T_prices'): 0.6355603053963895, ('MSFT_prices', 'UTX_prices'): 0.5620745895937173, ('ABT_prices', 'AFG_prices', 'WWD_prices'): 0.40085402905827167, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.49350863673722134, ('AAPL_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.4366049115989457, ('AAPL_prices', 'ABT_prices', 'AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.35476033538129476, ('MSFT_prices', 'WWD_prices'): 0.6486995094685285, ('ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.5545393842211289, ('MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.5376852114437556, ('AAPL_prices', 'ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'UTX_prices', 'WWD_prices'): 0.35131708377367754, ('AAPL_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'UTX_prices', 'WWD_prices'): 0.3757889830073739, ('MSFT_prices', 'T_prices', 'UTX_prices'): 0.4581259146998447, ('ABT_prices', 'WWD_prices'): 0.45543664229054137, ('ABT_prices', 'MCD_prices', 'T_prices'): 0.588413079946708, ('AAPL_prices', 'ABT_prices', 'T_prices', 'UTX_prices'): 0.5255165174263912, ('AAPL_prices', 'AFG_prices', 'T_prices', 'UTX_prices'): 0.35713142021479355, ('T_prices', 'UTX_prices'): 0.48596863956214426, ('AFG_prices', 'T_prices', 'UTX_prices'): 0.456669517282395, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.45146382357763093, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.3923641233044222, ('MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.5810208326631378, ('ABT_prices', 'UTX_prices'): 0.4592804076535704}
msemodelmultiurfstat['CAT_prices'] = {('ABT_prices', 'MSFT_prices', 'WWD_prices'): 0.17736976009336125, ('ABT_prices', 'T_prices', 'UTX_prices'): 0.11042730482977718, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.1676416098447602, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices'): 0.1072822023092951, ('AAPL_prices', 'ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.31808422403281533, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.17328124361957964, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.10217315082967056, ('AAPL_prices', 'ABT_prices', 'MCD_prices', 'T_prices', 'UTX_prices', 'WWD_prices'): 0.10556307614021236, ('AAPL_prices', 'MCD_prices'): 0.48831933091556695, ('ABT_prices', 'T_prices'): 0.17101878478653793, ('MSFT_prices', 'UTX_prices'): 0.35524829777033784, ('ABT_prices', 'AFG_prices', 'WWD_prices'): 0.13616580918620402, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.17732657427131118, ('AAPL_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.13033230645500837, ('AAPL_prices', 'ABT_prices', 'AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.10599772345662907, ('MSFT_prices', 'WWD_prices'): 0.17427348047625232, ('ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.06108438226286837, ('MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.08059663962678149, ('AAPL_prices', 'ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'UTX_prices', 'WWD_prices'): 0.06965758388641707, ('AAPL_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'UTX_prices', 'WWD_prices'): 0.16183531540725996, ('MSFT_prices', 'T_prices', 'UTX_prices'): 0.22689979230268545, ('ABT_prices', 'WWD_prices'): 0.1490116866633351, ('ABT_prices', 'MCD_prices', 'T_prices'): 0.08012606628994652, ('AAPL_prices', 'ABT_prices', 'T_prices', 'UTX_prices'): 0.08913135698101883, ('AAPL_prices', 'AFG_prices', 'T_prices', 'UTX_prices'): 0.2787155972099435, ('T_prices', 'UTX_prices'): 0.5314837129293156, ('AFG_prices', 'T_prices', 'UTX_prices'): 0.06415826143620455, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.4368803779844436, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.4042402030551931, ('MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.11648235829201198, ('ABT_prices', 'UTX_prices'): 0.13939548869513557}
msemodelmultiurrmse['MCD_prices'] = {('T_prices', 'WWD_prices'): 0.543953578225043, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.40379399354778783, ('AFG_prices', 'T_prices'): 0.5583119566881392}
msemodelmultiurfstat['MCD_prices'] = {'MCD_prices': {('T_prices', 'WWD_prices'): 0.5967801913707823, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.3826653333367738, ('AFG_prices', 'T_prices'): 0.5557150233488058}}
msemodelmultiurrmse['UTX_prices'] = {('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices'): 0.34199860828226153, ('ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.41760508824599685, ('AFG_prices', 'MCD_prices', 'T_prices'): 0.31553585153604147, ('MCD_prices', 'T_prices'): 0.6955822481626974, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.28688089447557535, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.5297617712181018, ('MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.511193973645606, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.3644740872664715, ('ABT_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.4557720568229035, ('ABT_prices', 'T_prices'): 0.6000092971168562, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.3538900902119019, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.365983354346552, ('MSFT_prices', 'T_prices', 'WWD_prices'): 0.5698645073404045, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.27849367280848264, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.36076814865168516, ('ABT_prices', 'MSFT_prices', 'ORCL_prices'): 0.6357018313008664, ('AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.374153132691391, ('MCD_prices', 'MSFT_prices', 'T_prices'): 0.5234724068847283, ('MSFT_prices', 'WWD_prices'): 0.6479669422793692, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.5649838153364137, ('MCD_prices', 'T_prices', 'WWD_prices'): 0.6435734323126007, ('MSFT_prices', 'ORCL_prices'): 0.6731226724238522, ('ABT_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.5479250264533684, ('AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.3463611167314072, ('ABT_prices', 'MSFT_prices', 'T_prices'): 0.5656134906098146, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices'): 0.36331426312792053, ('ABT_prices', 'MCD_prices', 'T_prices'): 0.5036514780366838, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.3353324547522711, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.4318037060647716, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.5714856452175674, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.38188912686369236, ('T_prices', 'WWD_prices'): 0.6547799359143228, ('AFG_prices', 'T_prices'): 0.3537948872372479, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.5561180829229674, ('AFG_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.4684475883247636, ('ABT_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.4157293836335451, ('ABT_prices', 'AFG_prices', 'T_prices'): 0.48096714766594784, ('ABT_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.44752129824673026}
msemodelmultiurfstat['UTX_prices'] = {('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices'): 0.4726711332725815, ('ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.5411053419289126, ('AFG_prices', 'MCD_prices', 'T_prices'): 1.2044475921724094, ('MCD_prices', 'T_prices'): 0.07140188550599753, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.27573275667443, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.06648657194973615, ('MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.11477156758399663, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.43624039451125957, ('ABT_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.24100080762430653, ('ABT_prices', 'T_prices'): 0.24206097436665575, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.4444992606023919, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.4971310032164322, ('MSFT_prices', 'T_prices', 'WWD_prices'): 0.14900985669422415, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.7270307896607195, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.15234503153138385, ('ABT_prices', 'MSFT_prices', 'ORCL_prices'): 0.058865397707616625, ('AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.5230782734363477, ('MCD_prices', 'MSFT_prices', 'T_prices'): 0.3287849350116151, ('MSFT_prices', 'WWD_prices'): 0.16467225862029175, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.12516821552196924, ('MCD_prices', 'T_prices', 'WWD_prices'): 0.08081255881432144, ('MSFT_prices', 'ORCL_prices'): 0.12114649096323533, ('ABT_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.16019856843492505, ('AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.8580995418480327, ('ABT_prices', 'MSFT_prices', 'T_prices'): 0.06081150304593676, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices'): 0.5568160901259905, ('ABT_prices', 'MCD_prices', 'T_prices'): 0.19131844793902145, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.3345600519858401, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.11385599732162006, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.14575045129094166, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.19346696400073773, ('T_prices', 'WWD_prices'): 0.13816580400461378, ('AFG_prices', 'T_prices'): 0.11797759938964868, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.16516071348308345, ('AFG_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.21995642513879266, ('ABT_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.36053286795909645, ('ABT_prices', 'AFG_prices', 'T_prices'): 0.24750578085966954, ('ABT_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.2243551951605282}
msemodelmultiurrmse['WWD_prices'] = {('AFG_prices', 'ORCL_prices', 'T_prices'): 0.5528706107972726, ('AFG_prices', 'T_prices'): 0.6914610887895177}
msemodelmultiurfstat['WWD_prices'] = {('AFG_prices', 'ORCL_prices', 'T_prices'): 0.25067434456750975, ('AFG_prices', 'T_prices'): 0.6509470443574962}
# TO DO : Update APA results
# Multi-cause RNN outputs
# msemodelmultiurrmse['AEM_prices'] = {('ABT_prices', 'MSFT_prices', 'WWD_prices'): 0.6330247113223922, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.5603878050307727, ('MSFT_prices', 'ORCL_prices', 'T_prices'): 0.3747723830553521, ('ABT_prices', 'T_prices', 'WWD_prices'): 0.6348235180081107, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.5257065877358602, ('AFG_prices', 'ORCL_prices', 'WWD_prices'): 0.564918402024877, ('AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.6042636538880495, ('AFG_prices', 'WWD_prices'): 0.6013030854705503, ('MSFT_prices', 'T_prices'): 0.5947379578803135, ('AFG_prices', 'MSFT_prices'): 0.7258896513444317, ('MSFT_prices', 'WWD_prices'): 0.6228128973972875, ('MSFT_prices', 'ORCL_prices'): 0.5461912012158421, ('ABT_prices', 'MSFT_prices'): 0.7102550590346185, ('ABT_prices', 'ORCL_prices'): 0.7203443436255015, ('ABT_prices', 'WWD_prices'): 0.6160958118521564, ('ABT_prices', 'MSFT_prices', 'T_prices'): 0.6629322260536813, ('MSFT_prices', 'T_prices', 'WWD_prices'): 0.6483101068960024, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.4080313760639106, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.6033536413327425, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.6549296692107568, ('ORCL_prices', 'T_prices'): 0.3726434880698962, ('T_prices', 'WWD_prices'): 0.7617414703868899, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.5893119624395394, ('AFG_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.43503158941718123, ('ORCL_prices', 'WWD_prices'): 0.7385465671928116, ('ABT_prices', 'ORCL_prices', 'WWD_prices'): 0.6179427421256892, ('ABT_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.5882109212042081, ('MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.5884108816552218}
# msemodelmultiurfstat['AEM_prices'] = {('ABT_prices', 'MSFT_prices', 'WWD_prices'): 0.1220021056538087, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.07829550975840283, ('MSFT_prices', 'ORCL_prices', 'T_prices'): 0.5869311207823803, ('ABT_prices', 'T_prices', 'WWD_prices'): 0.19992635555943233, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.14770036253739288, ('AFG_prices', 'ORCL_prices', 'WWD_prices'): 0.30735087500351727, ('AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.07289277242564939, ('AFG_prices', 'WWD_prices'): 0.47414151850823616, ('MSFT_prices', 'T_prices'): 0.40164168840602815, ('AFG_prices', 'MSFT_prices'): 0.22113029419487998, ('MSFT_prices', 'WWD_prices'): 0.5294429200194336, ('MSFT_prices', 'ORCL_prices'): 0.5088831672296618, ('ABT_prices', 'MSFT_prices'): 0.3603949237398955, ('ABT_prices', 'ORCL_prices'): 0.14408992990162672, ('ABT_prices', 'WWD_prices'): 0.5461179220767935, ('ABT_prices', 'MSFT_prices', 'T_prices'): 0.07138411910164888, ('MSFT_prices', 'T_prices', 'WWD_prices'): 0.1749646693524143, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.28839745807567, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.05215826095928522, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.16308896389569577, ('ORCL_prices', 'T_prices'): 1.2116009966312649, ('T_prices', 'WWD_prices'): 0.2504987760969164, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.23175787631988637, ('AFG_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.50547611976449, ('ORCL_prices', 'WWD_prices'): 0.28977212640999006, ('ABT_prices', 'ORCL_prices', 'WWD_prices'): 0.16571373772844447, ('ABT_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.12703148166052544, ('MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.0584659747374004}
#
# msemodelmultiurrmse['MCD_prices'] = {('MSFT_prices', 'WWD_prices'): 0.7008607477540778, ('ORCL_prices', 'T_prices', 'WWD_prices'): 0.544225584440246, ('T_prices', 'WWD_prices'): 0.635816398863065, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.398700764772724, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.37129658303847723, ('AFG_prices', 'T_prices'): 0.536089791302953}
# msemodelmultiurfstat['MCD_prices'] = {('MSFT_prices', 'WWD_prices'): 0.1504064956342399, ('ORCL_prices', 'T_prices', 'WWD_prices'): 0.16829567929450262, ('T_prices', 'WWD_prices'): 0.25017490644648827, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.34459183094004464, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.07380671674914624, ('AFG_prices', 'T_prices'): 0.3797328407903818}
#
# msemodelmultiurrmse['UTX_prices'] = {('ABT_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.47457824026937995, ('MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.6461731299781158, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.41324700325121483, ('MCD_prices', 'MSFT_prices'): 0.7881371158982571, ('AFG_prices', 'MCD_prices'): 0.5383803852296936, ('AFG_prices', 'MCD_prices', 'ORCL_prices', 'WWD_prices'): 0.6242316679094269, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'WWD_prices'): 0.5260888575929946, ('ABT_prices', 'MCD_prices', 'WWD_prices'): 0.5831463683989441, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices'): 0.369111607349844, ('AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices'): 0.30372525140963547, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.2810866000962978, ('MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.6458872348259599, ('ABT_prices', 'MCD_prices'): 0.8382802579134777, ('ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.5158102984622143, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'WWD_prices'): 0.48139895793970783, ('ABT_prices', 'T_prices'): 0.5634217978777495, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.5228617289173423, ('AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.5998576365946021, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.49961296993297954, ('AFG_prices', 'ORCL_prices'): 0.5266817310479294, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.3496271246022839, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'WWD_prices'): 0.5331406552209368, ('AFG_prices', 'MSFT_prices', 'T_prices'): 0.49581627037202, ('ABT_prices', 'AFG_prices', 'WWD_prices'): 0.5182120503361377, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.43091202757822983, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.29873083765798314, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.33846705717044234, ('MCD_prices', 'ORCL_prices', 'T_prices'): 0.6814644181612322, ('AFG_prices', 'T_prices'): 0.5682204002275811, ('MCD_prices', 'ORCL_prices'): 0.7936865995734962, ('ABT_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.4145788837575553, ('ABT_prices', 'ORCL_prices'): 0.6318676810996348, ('ABT_prices', 'WWD_prices'): 0.5606789393172547, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.33428356700162587, ('MCD_prices', 'T_prices'): 0.7082202181525546, ('MCD_prices', 'MSFT_prices', 'WWD_prices'): 0.6561964901744632, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.5218023453661799, ('MCD_prices', 'MSFT_prices', 'ORCL_prices'): 0.7222677062314797, ('MCD_prices', 'WWD_prices'): 0.5754259442231947, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices'): 0.6360214795025488, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.39895022840453476, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.2667353700815184, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.2949039771216216, ('AFG_prices', 'WWD_prices'): 0.5831877839769677, ('MCD_prices', 'MSFT_prices', 'T_prices'): 0.496622843296174, ('AFG_prices', 'MCD_prices', 'ORCL_prices'): 0.6752498631605077, ('MCD_prices', 'T_prices', 'WWD_prices'): 0.6663674859089683, ('ABT_prices', 'T_prices', 'WWD_prices'): 0.5883706562572791, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.3167523309666535, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices'): 0.5893020824489605, ('AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.5320970332985882, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.527026485629219, ('ABT_prices', 'MSFT_prices', 'T_prices'): 0.5225828024587104, ('ABT_prices', 'MCD_prices', 'T_prices'): 0.5169570807801733, ('ABT_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.5624720248187424, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.3786304694842773, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.3661974586787397, ('ABT_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.44956025125923377, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.5872805193859718, ('ABT_prices', 'MCD_prices', 'ORCL_prices'): 0.5495867796994793, ('ORCL_prices', 'WWD_prices'): 0.582843937910535, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.47575180119542, ('ABT_prices', 'AFG_prices', 'T_prices'): 0.4568360377508608, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.30354518122912744, ('MSFT_prices', 'WWD_prices'): 0.6226923589852239, ('MCD_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.6332036358531495, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices'): 0.30975403137066915, ('AFG_prices', 'MCD_prices', 'T_prices'): 0.33237224636184926, ('T_prices', 'WWD_prices'): 0.6324114387940986, ('ABT_prices', 'AFG_prices', 'ORCL_prices'): 0.41257891266821906, ('MCD_prices', 'ORCL_prices', 'WWD_prices'): 0.678352949346183, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.3359696930523303, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.39929158270957965, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.5601155574471537, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.3075354341613858, ('AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.3267941662191754, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.35009156750852666, ('AFG_prices', 'MCD_prices', 'MSFT_prices'): 0.6036967689595777, ('MSFT_prices', 'ORCL_prices'): 0.699215810619156, ('ABT_prices', 'MCD_prices', 'ORCL_prices', 'T_prices'): 0.48251479257453117, ('AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.40010452117010387, ('MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.67208024899503, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'WWD_prices'): 0.5455225680917745, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.5104408908316702, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices'): 0.2986790147377816, ('ABT_prices', 'AFG_prices', 'MCD_prices'): 0.6291597983113072}
# msemodelmultiurfstat['UTX_prices'] = {('ABT_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.41616322023855135, ('MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.05461583985132832, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.3611036992246812, ('MCD_prices', 'MSFT_prices'): 0.23724168760974101, ('AFG_prices', 'MCD_prices'): 0.5110523722254844, ('AFG_prices', 'MCD_prices', 'ORCL_prices', 'WWD_prices'): 0.08670063410594037, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'WWD_prices'): 0.2473111351887357, ('ABT_prices', 'MCD_prices', 'WWD_prices'): 0.43751260976727985, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices'): 0.41578533986173555, ('AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices'): 1.2436870658545882, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.09409496595009093, ('MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.08256638762579976, ('ABT_prices', 'MCD_prices'): 0.1632339973852507, ('ABT_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.14067256511044687, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'WWD_prices'): 0.07647106789342176, ('ABT_prices', 'T_prices'): 0.3579132634425952, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.0712498667801727, ('AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.16563625761040784, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices'): 0.2006446443435403, ('AFG_prices', 'ORCL_prices'): 0.5446158662886412, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.18196493970912958, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'WWD_prices'): 0.17085737468424986, ('AFG_prices', 'MSFT_prices', 'T_prices'): 0.146030161134557, ('ABT_prices', 'AFG_prices', 'WWD_prices'): 0.08194886427973036, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.18455939533737506, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.588648309595424, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.17970589530226516, ('MCD_prices', 'ORCL_prices', 'T_prices'): 0.16467797645997218, ('AFG_prices', 'T_prices'): 0.43169966783165603, ('MCD_prices', 'ORCL_prices'): 0.22859085168608403, ('ABT_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.2605147607187719, ('ABT_prices', 'ORCL_prices'): 0.19704610815592444, ('ABT_prices', 'WWD_prices'): 0.5592756098469369, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 1.0105093858585075, ('MCD_prices', 'T_prices'): 0.37685436019533136, ('MCD_prices', 'MSFT_prices', 'WWD_prices'): 0.20106877695843012, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.07342476020127345, ('MCD_prices', 'MSFT_prices', 'ORCL_prices'): 0.09119805454193579, ('MCD_prices', 'WWD_prices'): 0.6945987665853456, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices'): 0.06167776548779542, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.2929189200493041, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.3125052271902799, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.5491009724919942, ('AFG_prices', 'WWD_prices'): 0.39495541678412044, ('MCD_prices', 'MSFT_prices', 'T_prices'): 0.5869932817976133, ('AFG_prices', 'MCD_prices', 'ORCL_prices'): 0.1753969054634757, ('MCD_prices', 'T_prices', 'WWD_prices'): 0.06280728446181087, ('ABT_prices', 'T_prices', 'WWD_prices'): 0.07485210567258735, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.7938949288660523, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices'): 0.22563236707047502, ('AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.21438964989589307, ('AFG_prices', 'T_prices', 'WWD_prices'): 0.1999613985985163, ('ABT_prices', 'MSFT_prices', 'T_prices'): 0.07814837232854757, ('ABT_prices', 'MCD_prices', 'T_prices'): 0.36997875545825537, ('ABT_prices', 'MCD_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.14880936556150734, ('ABT_prices', 'AFG_prices', 'T_prices', 'WWD_prices'): 0.5539442904811213, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.22764437765699286, ('ABT_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.16243106679235522, ('AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.06029799802703615, ('ABT_prices', 'MCD_prices', 'ORCL_prices'): 0.14971412057099992, ('ORCL_prices', 'WWD_prices'): 0.49997784684249635, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'WWD_prices'): 0.23442626577621772, ('ABT_prices', 'AFG_prices', 'T_prices'): 0.23331294232311878, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.24736116037524128, ('MSFT_prices', 'WWD_prices'): 0.40398863486454223, ('MCD_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.07130299154426253, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'T_prices'): 0.6689276923778058, ('AFG_prices', 'MCD_prices', 'T_prices'): 1.1308043192677546, ('T_prices', 'WWD_prices'): 0.38241173609904694, ('ABT_prices', 'AFG_prices', 'ORCL_prices'): 0.5315074563875245, ('MCD_prices', 'ORCL_prices', 'WWD_prices'): 0.17002012055593668, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices'): 0.4781775069777514, ('AFG_prices', 'MCD_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.317555591888446, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'WWD_prices'): 0.15313211039830593, ('ABT_prices', 'AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.34806867016174853, ('AFG_prices', 'MSFT_prices', 'T_prices', 'WWD_prices'): 0.5172127339613654, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.4162473946475278, ('AFG_prices', 'MCD_prices', 'MSFT_prices'): 0.3055181946004901, ('MSFT_prices', 'ORCL_prices'): 0.2682454638614982, ('ABT_prices', 'MCD_prices', 'ORCL_prices', 'T_prices'): 0.4123181893039485, ('AFG_prices', 'MCD_prices', 'T_prices', 'WWD_prices'): 0.6654835190569195, ('MCD_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices'): 0.07467479859956547, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'WWD_prices'): 0.06896836631118086, ('AFG_prices', 'MSFT_prices', 'ORCL_prices', 'T_prices', 'WWD_prices'): 0.09731717718489927, ('ABT_prices', 'AFG_prices', 'MCD_prices', 'ORCL_prices', 'T_prices'): 0.6154961305137022, ('ABT_prices', 'AFG_prices', 'MCD_prices'): 0.3323805178961833}
#
# msemodelmultiurrmse['WWD_prices'] = {('AFG_prices', 'ORCL_prices'): 0.7145028137682614, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.4868776333551425, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.5121672723205714}
# msemodelmultiurfstat['WWD_prices'] = {('AFG_prices', 'ORCL_prices'): 0.077952487254866, ('ABT_prices', 'AFG_prices', 'ORCL_prices', 'T_prices'): 0.051942494854722285, ('AFG_prices', 'ORCL_prices', 'T_prices'): 0.39505753761057566}
# TO DO : Update APA, CAT results
# print('msemodelmultiurrmse',msemodelmultiurrmse.keys())
# print('msemodelmultiurfstat',msemodelmultiurfstat.keys())
# for stock in msemodelmultiurrmse.keys():
#     topgrumulticauses = sorted(msemodelmultiurrmse[stock].items(), key=lambda t: t[1], reverse=False)
#     # print('topgrumulticauses',topgrumulticauses)
#     print('top5multicauses',stock,topgrumulticauses[0:5])
#     print('\n')
# os.chdir("/content/drive/My Drive/models/")
# for file in glob.glob('grudnnmsemodelmultiurrmse*.pkl'):
#     print file
handle = open(MsemodelurrmsePath, 'rb')
msemodelurrmse = pickle.load(handle)
handle = open(GrurnnmodeluroutputPath, 'rb')
grurnnmodelurrmse = pickle.load(handle)
for stock in msemodelmultiurrmse.keys():
    # print('bottom5onecauses',sorted(msemodelurrmse[stock].items(), key=lambda t: t[1], reverse=True))
    print('bottom5onecauses',sorted(grurnnmodelurrmse[stock].items(), key=lambda t: t[1], reverse=True))
    print('\n')
sys.exit()
# multivarqmemodelrrmse = collections.defaultdict(dict)
#
# qmemodelgraph = nx.read_gpickle(QmemodelurrmsegraphPath)
#
# for stock in sorted(qmemodelgraph.nodes()):
#     stockcauses = [col + '_prices' for col in qmemodelgraph.predecessors(stock)]
# # msemodelgraph = nx.read_gpickle(MsemodelurrmsegraphPath)
# # for stock in sorted(msemodelgraph.nodes()):
# #     stockcauses = [col + '_prices' for col in msemodelgraph.predecessors(stock)]
#     if stockcauses:
#         # print('stock, stockcauses',stock, stockcauses)
#
#         Ypast, Ycurr = getstockdata(df[stockcauses], df[stock + '_prices'])
#
#         numrecords = len(Ycurr)
#         numtestrecords = int(math.ceil(0.3 * numrecords))
#         numtrainrecords = int(math.ceil(0.7 * numrecords))
#
#         modelr = restricted_qme_model(q)
#
#         np.random.seed(3)
#         modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2,
#                    validation_split=0.1)
#         Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
#
#         multivarqmemodelrrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
#
# print('multivarqmemodelrrmse',multivarqmemodelrrmse)
# with open(MultivarQmemodelrrmsePath, 'wb') as handle:
#     pickle.dump(multivarqmemodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# multivarqmemodelrrmse = pickle.load(open(MultivarQmemodelrrmsePath, 'rb'))
# print('multivarqmemodelrrmse',multivarqmemodelrrmse)
# ('multivarqmemodelrrmse', defaultdict(<type 'dict'>, {'AFG': {'AAPL_prices,MCD_prices,CAT_prices,WWD_prices,ORCL_prices,AEM_prices': 1.102662920940748}, 'AAPL': {'WWD_prices,UTX_prices,MCD_prices': 2.211063246457727}, 'MCD': {'AAPL_prices,AFG_prices,CAT_prices,WWD_prices,UTX_prices,AEM_prices': 0.35771888827819764}, 'CAT': {'WWD_prices,UTX_prices,AFG_prices,AEM_prices': 0.7458992355955458}, 'WWD': {'AFG_prices,AAPL_prices,MCD_prices,UTX_prices,APA_prices,AEM_prices': 0.3623143314380908}, 'ORCL': {'AFG_prices,MCD_prices': 0.961790211980454}, 'ABT': {'AFG_prices,AAPL_prices,MCD_prices,CAT_prices,WWD_prices,ORCL_prices,T_prices,UTX_prices,APA_prices,AEM_prices,MSFT_prices': 0.38642372075477355}, 'UTX': {'WWD_prices,AAPL_prices,CAT_prices': 0.3731779445061958}, 'MSFT': {'WWD_prices,AAPL_prices,AFG_prices,ABT_prices,MCD_prices': 0.6271186407369729}}))
# multivarmsemodelrrmse = collections.defaultdict(dict)
# multivargrumodelrrmse = collections.defaultdict(dict)
# multivarlstmmodelrrmse = collections.defaultdict(dict)
#
# msemodelgraph = nx.read_gpickle(MsemodelurrmsegraphPath)
#
# for stock in sorted(msemodelgraph.nodes()):
#     stockcauses = [col + '_prices' for col in msemodelgraph.predecessors(stock)]
# # qmemodelgraph = nx.read_gpickle(QmemodelurrmsegraphPath)
# # for stock in sorted(qmemodelgraph.nodes()):
# #     stockcauses = [col + '_prices' for col in qmemodelgraph.predecessors(stock)]
#     if stockcauses:
#         # print('stock, stockcauses',stock, stockcauses)
#
#         # Ypast, Ycurr = getstockdata(df[stockcauses], df[stock + '_prices'])
#         # modelr = restricted_mse_model(p)
#
#         n_lag = 200
#         nb_epoch = 15
#         n_neurons = 25
#
#         raw_values = df[stock + '_prices']
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         cols = []
#         for col in df[stockcauses]:
#             cols.append(preprocess(df[col],scaler))
#
#         # indata = np.concatenate(cols, axis=1)
#         outdata = preprocess(df[stock + '_prices'],scaler)
#         Ycurr = outdata.values
#
#         indata = series_to_supervised(concat(cols, axis=1), n_lag)
#         Ypast = indata.values
#
#         numrecords = len(Ycurr)
#         numtestrecords = int(math.ceil(0.3 * numrecords))
#         numtrainrecords = int(math.ceil(0.7 * numrecords))
#
#         # np.random.seed(3)
#         # modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2,
#         #            validation_split=0.1)
#         # Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
#         # multivarmsemodelrrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
#
#         # modelr = fit_rnn(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], n_lag, 1, nb_epoch, n_neurons, "GRU")
#         modelr = fit_rnn(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], n_lag, 1, nb_epoch, n_neurons, "LSTM")
#
#         forecasts = make_forecasts(modelr, 1, Ypast[-numtestrecords:])
#         Ycurrp = inverse_transform(Series(raw_values), forecasts, scaler, numtestrecords + 2)
#
#         # multivargrumodelrrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, raw_values[-numtestrecords:]))
#         # print('multivargrumodelrrmse', multivargrumodelrrmse)
#         multivarlstmmodelrrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, raw_values[-numtestrecords:]))
#
# # print('multivarmsemodelrrmse',multivarmsemodelrrmse)
# # with open(MultivarMsemodelrrmsePath, 'wb') as handle:
# #     pickle.dump(multivarmsemodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # print('multivargrumodelrrmse',multivargrumodelrrmse)
# # with open(MultivarGrumodelrrmsePath, 'wb') as handle:
# #     pickle.dump(multivargrumodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # ('multivargrumodelrrmse', defaultdict(<type 'dict'>, {'MCD': {'WWD_prices,AAPL_prices,AFG_prices,UTX_prices': 0.02832811622294217}, 'AFG': {'WWD_prices,AAPL_prices': 0.028929579586017475}, 'ORCL': {'UTX_prices': 0.04037686659146498}, 'ABT': {'AFG_prices,AAPL_prices,MCD_prices,CAT_prices,WWD_prices,ORCL_prices,T_prices,UTX_prices,APA_prices,MSFT_prices': 0.03086369912855997}, 'UTX': {'WWD_prices': 0.031563500584780566}, 'MSFT': {'WWD_prices,UTX_prices,AFG_prices,ABT_prices': 0.03278140161539794}}))
#
#
# print('multivarlstmmodelrrmse',multivarlstmmodelrrmse)
# with open(MultivarLstmmodelrrmsePath, 'wb') as handle:
#     pickle.dump(multivarlstmmodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # ('multivarlstmmodelrrmse', defaultdict(<type 'dict'>, {'MCD': {'WWD_prices,AAPL_prices,AFG_prices,UTX_prices': 0.02854622163480819}, 'AFG': {'WWD_prices,AAPL_prices': 0.029986510360138576}, 'ORCL': {'UTX_prices': 0.04128145927765118}, 'ABT': {'AFG_prices,AAPL_prices,MCD_prices,CAT_prices,WWD_prices,ORCL_prices,T_prices,UTX_prices,APA_prices,MSFT_prices': 0.03090224678608855}, 'UTX': {'WWD_prices': 0.03290983013604868}, 'MSFT': {'WWD_prices,UTX_prices,AFG_prices,ABT_prices': 0.03263745370568845}}))
# mse features : ('multivarmsemodelrrmse', defaultdict(<type 'dict'>, {'MCD': {'WWD_prices,AAPL_prices,AFG_prices,UTX_prices': 0.27199359590745903}, 'AFG': {'WWD_prices,AAPL_prices': 1.0651366154941102}, 'ORCL': {'UTX_prices': 0.8489948861211455}, 'ABT': {'AFG_prices,AAPL_prices,MCD_prices,CAT_prices,WWD_prices,ORCL_prices,T_prices,UTX_prices,APA_prices,MSFT_prices': 0.25509750781876983}, 'UTX': {'WWD_prices': 0.5909489390706406}, 'MSFT': {'WWD_prices,UTX_prices,AFG_prices,ABT_prices': 0.6317953485497307}}))
# qme features :
# ['AAPL_prices', 'APA_prices', 'T_prices', 'ABT_prices', 'WWD_prices', 'AEM_prices', 'UTX_prices', 'AFG_prices', 'MSFT_prices', 'ORCL_prices', 'MCD_prices', 'CAT_prices']
# ['AAPL_prices', 'APA_prices', 'T_prices',  'WWD_prices', 'AEM_prices',  'CAT_prices']
handle = open(MsemodelurrmsePath, 'rb')
msemodelurrmse = pickle.load(handle)
msemodelgraph = nx.read_gpickle(MsemodelurrmsegraphPath)
print('msemodelurrmse', msemodelurrmse.keys())
print(msemodelurrmse.keys())
print(sorted(msemodelgraph.nodes()))
handle = open(GrurnnmodeluroutputPath, 'rb')
grurnnmodelurrmse = pickle.load(handle)
grurnnmodelgraph = nx.read_gpickle(GrurnnmodelurrmsegraphPath)
print('grurnnmodelurrmse', grurnnmodelurrmse.keys())
msemodelmultiurrmse = collections.defaultdict(dict)
msemodelmultiurfstat = collections.defaultdict(dict)
# grurnnmodelmultiurrmse = collections.defaultdict(dict)
# grurnnmodelmultiurfstat = collections.defaultdict(dict)
# effects = []
# for effect in sorted(msemodelgraph.nodes()):
#     predecessors = list(msemodelgraph.predecessors(effect))
#     if(predecessors):
#         effects.append(effect + '_prices')
# #     print(list(msemodelgraph.predecessors(effect)))
# # print('effects',effects)
# # sys.exit()
# # ('effects', ['ABT_prices', 'AFG_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'UTX_prices'])
#
#
effects = []
for effect in sorted(grurnnmodelgraph.nodes()):
    predecessors = list(grurnnmodelgraph.predecessors(effect))
    if(predecessors):
        effects.append(effect + '_prices')
    print(list(grurnnmodelgraph.predecessors(effect)))
print('effects',effects)
# sys.exit()
# effects1 = ['AAPL_prices']
# effects2 = ['AEM_prices']
# effects3 = ['AFG_prices']
# effects4 = ['APA_prices']
effects5 = ['CAT_prices']
# effects6 = ['MCD_prices']
# effects7 = ['MSFT_prices']
# effects8 = ['ORCL_prices']
# effects9 = ['T_prices']
# effects10 = ['UTX_prices']
# effects11 = ['WWD_prices']
#
# # for effect in effects:
# for effect in effects1:
# for effect in effects2:
# for effect in effects3:
# for effect in effects4:
for effect in effects5:
# for effect in effects6:
# for effect in effects7:
# for effect in effects8:
# for effect in effects9:
# for effect in effects10:
# for effect in effects11:
# for effect in ['MSFT_prices']:
# # for effect in msemodelurrmse.keys():
# # for effect in ['ABT_prices']:
# #     onecauses = zip(msemodelurrmse[effect].keys())
#
# # for effect in sorted(msemodelgraph.nodes()):
#     onecauses = zip([col + '_prices' for col in msemodelgraph.predecessors(effect.rstrip('_prices'))])
    onecauses = zip([col + '_prices' for col in grurnnmodelgraph.predecessors(effect.rstrip('_prices'))])
    print('onecauses',onecauses)
    if onecauses:
        maximalcauses = onecauses
        firstiter = 0
        while maximalcauses:
            firstiter += 1
            candidatecauseslist = list(prod for prod in itertools.product(onecauses, maximalcauses) if prod[0][0] not in prod[1])
            # for prod in itertools.product(onecauses, maximalcauses):
            #     print(prod[0])
            #     print(prod[1])
            #     print(prod[0][0] in prod[1])
            #     print(prod[0] == prod[1])
            #     sys.exit()
            #
            # print('candidatecauseslist', candidatecauseslist)
            # print('onecauses', onecauses)
            # print('maximalcauses', maximalcauses)
            #
            # sys.exit()
            print('candidatecauseslist',candidatecauseslist)
            print('onecauses',onecauses)
            print('maximalcauses',maximalcauses)
            maximalcauses = []
            # processedcauses = []
            for candidatecauses in candidatecauseslist:
                # stockcauses = list(chain(candidatecauses))
                stockcauses = []
                for c in candidatecauses:
                    for uc in c:
                        stockcauses.append(uc)
                # print('candidatecauses',candidatecauses)
                stockcauses = filter(None, stockcauses)
                # alreadyprocessed = False
                # for processed in processedcauses:
                #     if(processed == set(stockcauses)):
                #         alreadyprocessed = True
                #
                # if(alreadyprocessed != True):
                # stockcauses = list(candidatecauses)
                # print('df[stockcauses]', df[stockcauses])
                if(firstiter == 1) :
                    # restrictederror = msemodelurrmse[effect][stockcauses[0]]
                    restrictederror = grurnnmodelurrmse[effect][stockcauses[0]]
                else:
                    # print(msemodelmultiurrmse[effect])
                    # print(grurnnmodelmultiurrmse[effect])
                    # print(maximalcauses)
                    restrictederror = msemodelmultiurrmse[effect][tuple(sorted(stockcauses[1:]))]
                    # restrictederror = grurnnmodelmultiurrmse[effect][tuple(sorted(stockcauses[1:]))]
                print('restrictederror',restrictederror)
                # print('stockcauses',stockcauses)
                Ypast, Ycurr = getstockdata(df[stockcauses], df[effect])
                  # TO DO : Continue rnn data creation from here. Use RNN preprocessing and data retrieval in validation code below. Use same parameters to fit_gru on onecause and multi-cause data.
                numrecords = len(Ycurr)
                numtestrecords = int(math.ceil(0.3 * numrecords))
                numtrainrecords = int(math.ceil(0.7 * numrecords))
                modelur = restricted_mse_model(q)
                np.random.seed(7)
                modelur.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=0, validation_split=0.1)
                Ycurrp = modelur.predict(Ypast[-numtestrecords:], batch_size=128)
                unrestrictederror = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
                # TO DO : Save unrestrictederror in dict
                print('unrestrictederror',unrestrictederror)
                fstat = (restrictederror - unrestrictederror) / unrestrictederror
                print('fstat',fstat)
                print('effect',effect)
                print('stockcauses',stockcauses)
                if ( fstat > 0.05 ):
                    founddup = False
                    for maximalcause in maximalcauses:
                        if sorted(list(maximalcause)) == sorted(stockcauses):
                            founddup = True
                    if(not founddup):
                        maximalcauses.append(tuple(sorted(stockcauses)))
                        msemodelmultiurrmse[effect][tuple(sorted(stockcauses))] = unrestrictederror
                        msemodelmultiurfstat[effect][tuple(sorted(stockcauses))] = fstat
                    print('maximalcauses',maximalcauses)
                    # processedcauses.append(set(stockcauses))
                    # TO DO : Print maximalcauses with more than 2 causes
            # TO DO : Track all the causes keys are correct. Unique keys identified by tuple order.
#
#
#                     # sys.exit()
#
            # print(maximalcauses)
            # print(msemodelmultiurrmse)
            # print(msemodelmultiurfstat)
#             # print("Here")
#             # sys.exit()
#
print('maximalcauses',maximalcauses)
print('msemodelmultiurrmse',msemodelmultiurrmse)
print('msemodelmultiurfstat',msemodelmultiurfstat)
# if(maximalcauses):
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grumsemodelmultiurrmse.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grumsemodelmultiurfstat.pkl"
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse1.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat1.pkl"
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse2.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat2.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse3.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat3.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse4.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat4.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse5.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat5.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse6.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat6.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse7.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat7.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse8.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat8.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse9.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat9.pkl"
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse10.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat10.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grudnnmsemodelmultiurrmse11.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grudnnmsemodelmultiurfstat11.pkl"
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse1.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat1.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse2.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat2.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse3.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat3.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse4.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat4.pkl"
#
MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse5.pkl"
MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat5.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse6.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat6.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse7.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat7.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse8.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat8.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse9.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat9.pkl"
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse10.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat10.pkl"
#
# MsemodelmultiurrmsePath = "/content/drive/My Drive/models/grurnnmsemodelmultiurrmse11.pkl"
# MsemodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmsemodelmultiurfstat11.pkl"
with open(MsemodelmultiurrmsePath, 'wb') as handle:
    pickle.dump(msemodelmultiurrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MsemodelmultiurfstatPath, 'wb') as handle:
    pickle.dump(msemodelmultiurfstat, handle, protocol=pickle.HIGHEST_PROTOCOL)
sys.exit()
GrurnnmodelmultiurfstatPath = "/content/drive/My Drive/models/grurnnmodelmultiurfstat.pkl"
# multivarmsemodelmvrmse = collections.defaultdict(dict)
multivargrumodelmvrmse = collections.defaultdict(dict)
# multivarlstmmodelmvrmse = collections.defaultdict(dict)
#
# multivarlinearmodelmvrmse = collections.defaultdict(dict)
# multivarsvmmodelmvrmse = collections.defaultdict(dict)
# multivarlinearmodelrrmse = collections.defaultdict(dict)
# multivarsvmmodelrrmse = collections.defaultdict(dict)
#
# handle = open(MsemodelmultiurfstatPath, 'rb')
# msemodelmultiurfstat = pickle.load(handle)
handle = open(GrurnnmodelmultiurfstatPath, 'rb')
grurnnmodelmultiurfstat = pickle.load(handle)
# for stock in msemodelmultiurfstat.keys():
for stock in grurnnmodelmultiurfstat.keys():
#     # tups = msemodelmultiurfstat[key].keys()
#     # tups.sort(key=lambda t: len(t), reverse=True)
#
#     multicauses = sorted(msemodelmultiurfstat[stock].items(), key=lambda t: len(t[0]), reverse=True)
    multicauses = sorted(grurnnmodelmultiurfstat[stock].items(), key=lambda t: len(t[0]), reverse=True)
#
    for multicause in multicauses:
        stockcauses = list(multicause[0])
        # # DNN
        # Ypast, Ycurr = getstockdata(df[stockcauses], df[stock])
        # numrecords = len(Ycurr)
        # numtestrecords = int(math.ceil(0.3 * numrecords))
        # numtrainrecords = int(math.ceil(0.7 * numrecords))
        # modelr = restricted_mse_model(p)
        # np.random.seed(3)
        # modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
        # Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
        # multivarmsemodelmvrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        # GRU
        n_lag = 200
        nb_epoch = 15
        n_neurons = 25
        raw_values = df[stock]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        cols = []
        for col in df[stockcauses]:
            cols.append(preprocess(df[col],scaler))
        indata = series_to_supervised(concat(cols, axis=1), n_lag)
        Ypast = indata.values
        outdata = preprocess(df[stock], scaler)
        Ycurr = outdata.values
        numrecords = len(Ycurr)
        numtestrecords = int(math.ceil(0.3 * numrecords))
        numtrainrecords = int(math.ceil(0.7 * numrecords))
        modelr = fit_rnn(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], n_lag, 1, nb_epoch, n_neurons, "GRU")
        forecasts = make_forecasts(modelr, 1, Ypast[-numtestrecords:])
        Ycurrp = inverse_transform(Series(raw_values), forecasts, scaler, numtestrecords + 2)
        multivargrumodelmvrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        # # # LSTM
        # n_lag = 200
        # nb_epoch = 15
        # n_neurons = 25
        # raw_values = df[stock]
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # cols = []
        # for col in df[stockcauses]:
        #     cols.append(preprocess(df[col],scaler))
        # indata = series_to_supervised(concat(cols, axis=1), n_lag)
        # Ypast = indata.values
        # outdata = preprocess(df[stock], scaler)
        # Ycurr = outdata.values
        # numrecords = len(Ycurr)
        # numtestrecords = int(math.ceil(0.3 * numrecords))
        # numtrainrecords = int(math.ceil(0.7 * numrecords))
        # modelr = fit_rnn(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], n_lag, 1, nb_epoch, n_neurons, "LSTM")
        # forecasts = make_forecasts(modelr, 1, Ypast[-numtestrecords:])
        # Ycurrp = inverse_transform(Series(raw_values), forecasts, scaler, numtestrecords + 2)
        # multivarlstmmodelmvrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        # # # Linear Regression
        # Ypast, Ycurr = getstockdata(df[['Date',stock]],p)
        # # Ypast, Ycurr = getstockdata(df[stockcauses], df[stock])
        # numrecords = len(Ycurr)
        # numtestrecords = int(math.ceil(0.3 * numrecords))
        # numtrainrecords = int(math.ceil(0.7 * numrecords))
        #
        # regr = linear_model.LinearRegression()
        # regr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords])
        # Ycurrp = regr.predict(Ypast[-numtestrecords:])
        # #
        # multivarlinearmodelrrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        # # multivarlinearmodelmvrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        # # Support Vector Regression
        # Ypast, Ycurr = getstockdata(df[['Date',stock]],p)
        # # Ypast, Ycurr = getstockdata(df[stockcauses], df[stock])
        # numrecords = len(Ycurr)
        # numtestrecords = int(math.ceil(0.3 * numrecords))
        # numtrainrecords = int(math.ceil(0.7 * numrecords))
        #
        # parameters = {'kernel':['linear','rbf'], 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma':[0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5]}
        # regr = svm.SVR()
        # grid = model_selection.GridSearchCV(regr,parameters,verbose=1,n_jobs=-1)
        # grid.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords])
        # Ycurrp = grid.predict(Ypast[-numtestrecords:])
        # multivarsvmmodelrrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        # # multivarsvmmodelmvrmse[stock][','.join(stockcauses)] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
# print('multivarmsemodelmvrmse',multivarmsemodelmvrmse)
# with open(MultivarMsemodelmultimvrmsePath, 'wb') as handle:
#     pickle.dump(multivarmsemodelmvrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('multivargrumodelmvrmse',multivargrumodelmvrmse)
# with open(MultivarGrumodelmultimvrmsePath, 'wb') as handle:
#     pickle.dump(multivargrumodelmvrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MultivarGrurnnmodelmultimvrmsePath, 'wb') as handle:
    pickle.dump(multivargrumodelmvrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('multivarlstmmodelmvrmse',multivarlstmmodelmvrmse)
# with open(MultivarLstmmodelmultimvrmsePath, 'wb') as handle:
#     pickle.dump(multivarlstmmodelmvrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('multivarlinearmodelrrmse',multivarlinearmodelrrmse)
# with open(MultivarLinearmodelmultirrmsePath, 'wb') as handle:
#     pickle.dump(multivarlinearmodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('multivarlinearmodelmvrmse',multivarlinearmodelmvrmse)
# with open(MultivarLinearmodelmultimvrmsePath, 'wb') as handle:
#     pickle.dump(multivarlinearmodelmvrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('multivarsvmmodelmvrmse',multivarsvmmodelmvrmse)
# with open(MultivarSvmmodelmultimvrmsePath, 'wb') as handle:
#     pickle.dump(multivarsvmmodelmvrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('multivarsvmmodelrrmse',multivarsvmmodelrrmse)
# with open(MultivarSvmmodelmultirrmsePath, 'wb') as handle:
#     pickle.dump(multivarsvmmodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
#sys.exit()
multivarmsemodelmvrmse = pickle.load(open(MultivarMsemodelmultimvrmsePath, 'rb'))
# multivargrumodelmvrmse = pickle.load(open(MultivarGrumodelmultimvrmsePath, 'rb'))
# multivargrumodelmvrmse = pickle.load(open(MultivarGrurnnmodelmultimvrmsePath, 'rb'))
# multivarlstmmodelmvrmse = pickle.load(open(MultivarLstmmodelmultimvrmsePath, 'rb'))

# multivarlinearmodelrrmse = pickle.load(open(MultivarLinearmodelmultirrmsePath, 'rb'))
# multivarlinearmodelmvrmse = pickle.load(open(MultivarLinearmodelmultimvrmsePath, 'rb'))
#
# multivarsvmmodelrrmse = pickle.load(open(MultivarSvmmodelmultirrmsePath, 'rb'))
# multivarsvmmodelmvrmse = pickle.load(open(MultivarSvmmodelmultimvrmsePath, 'rb'))


# print('multivarmsemodelmvrmse',multivarmsemodelmvrmse)

# stock = 'MCD_prices'
stock = 'ABT_prices'
topmsemulticauses = sorted(multivarmsemodelmvrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('topmsemulticauses',topmsemulticauses[0:10])
print('topmsemulticauses',topmsemulticauses)
# print('msemodelurrmseMCD_prices',msemodelurrmse['MCD_prices'])
# t = topmsemulticauses[0]
# print(t)
#
for tup in topmsemulticauses:
    l = len(set(tup[0].split(',')))
    if(l>7):
        print(tup,l)
#     tupint = set(t[0].split(',')).intersection(set(tup[0].split(',')))
#     if(len(tupint) == 1):
#         print(tup)


    # sys.exit()

# topgrumulticauses = sorted(multivargrumodelmvrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('topgrumulticauses',topgrumulticauses)
# print('topgrumulticauses',topgrumulticauses[0:5])

# toplstmmulticauses = sorted(multivarlstmmodelmvrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('toplstmmulticauses',toplstmmulticauses)
# print('toplstmmulticauses',toplstmmulticauses[0:5])

# toplinearrmulticauses = sorted(multivarlinearmodelrrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('toplinearrmulticauses',toplinearrmulticauses)
# print('toplinearrmulticauses',toplinearrmulticauses[0:5])
# toplinearmvmulticauses = sorted(multivarlinearmodelmvrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('toplinearmvmulticauses',toplinearmvmulticauses)
# print('toplinearmvmulticauses',toplinearmvmulticauses[0:5])

# topsvmrmulticauses = sorted(multivarsvmmodelrrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('topsvmrmulticauses',topsvmrmulticauses)
# print('topsvmrmulticauses',topsvmrmulticauses[0:5])
# topsvmmvmulticauses = sorted(multivarsvmmodelmvrmse[stock].items(), key=lambda t: t[1], reverse=False)
# print('topsvmmvmulticauses',topsvmmvmulticauses)
# print('topsvmmvmulticauses',topsvmmvmulticauses[0:5])
