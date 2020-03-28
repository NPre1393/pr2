import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from keras import backend as K

import tensorflow as tf
import math
import itertools

import collections
import pickle
from sklearn.metrics import mean_squared_error

QmemodelrmaePath = "/models/qmemodelrmae.pkl"
QmemodelrmsePath = "/models/qmemodelrmse.pkl"
QmemodelurmaePath = "/models/qmemodelurmae.pkl"
QmemodelurmsePath = "/models/qmemodelurmse.pkl"
QmemodelrrmsePath = "/models/qmemodelrrmse.pkl"
QmemodelurrmsePath = "/models/qmemodelurrmse.pkl"

MsemodelrmaePath = "/models/msemodelrmae.pkl"
MsemodelrmsePath = "/models/msemodelrmse.pkl"
MsemodelurmaePath = "/models/msemodelurmae.pkl"
MsemodelurmsePath = "/models/msemodelurmse.pkl"
MsemodelrrmsePath = "/models/msemodelrrmse.pkl"
MsemodelurrmsePath = "/models/msemodelurrmse.pkl"


df = pd.read_csv("pricesvolumes.csv")
# print(df.columns)
# sys.exit()

# cols = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,33,34,36,38,40,42]
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]

df.drop(df.columns[cols],axis=1,inplace=True)
# df['Date'] = pd.to_datetime(df['Date'])
df.fillna(0, inplace=True)

print(len(df.columns)) # 17

print((df.columns)) # Index([u'Date', u'^DJI_prices', u'^GSPC_prices', u'^IXIC_prices', u'AAPL_prices', u'ABT_prices', u'AEM_prices', u'AFG_prices', u'APA_prices', u'B_prices', u'CAT_prices', u'FRD_prices', u'GIGA_prices', u'LAKE_prices', u'MCD_prices', u'MSFT_prices', u'ORCL_prices', u'SUN_prices', u'T_prices', u'UTX_prices', u'WWD_prices'], dtype='object')
print(len(df.index)) # 5285

allcols = df.columns.tolist()
print('allcols',allcols[1:])
df[allcols[1:]] = df[allcols[1:]].apply(pd.to_numeric).apply(lambda x: x/x.mean(), axis=0)

allcols.remove("Date")
allcols.remove("IXIC_prices")
allcols.remove("B_prices")
allcols.remove("LAKE_prices")
allcols.remove("SUN_prices")

# allcols.remove("DJI_prices")
# allcols.remove("FRD_prices")
# allcols.remove("GSPC_prices")
# allcols.remove("GIGA_prices")


# print(len(df.columns))
# print(df['Date'])

# inputbatchsize + p, inputbatchsize + q < 5285
inputbatchsize = 5000
p = 200
q = 200

# inputbatchsize = 3000
# p = 2200
# q = 2200

percentilenum = 10
numepochs = 50

def getstockdata(dfone, lag):
    Ypast = []
    Ycurr = []
    for i in range(-inputbatchsize, 0):
        y = dfone.iloc[i,1]
        x = dfone.iloc[i - lag:i,1].tolist()
        Ypast.append(x)
        Ycurr.append(y)
    Ypast = np.vstack(Ypast)
    Ycurr = np.vstack(Ycurr)
    Ycurr = Ycurr.reshape(Ycurr.shape[0], )
    return Ypast,Ycurr

def restricted_mse_model(lag):
    model = Sequential()
    model.add(Dense(units=2*lag, activation='relu', kernel_initializer='normal', bias_initializer='zeros', input_dim=Ypast.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(lag/2, activation='linear', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(lag/2, activation='relu', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='normal', bias_initializer='zeros'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.compile(loss='mse', optimizer='adam', metrics=['msd'])
    return model

def quadratic_mean_error(y_true, y_pred):

    sumofsquares = 0
    currpercentile = 0
    prevpercentile = 0
    for i in range(10, 110, percentilenum):
        prevpercentile = currpercentile
        currpercentile = tf.contrib.distributions.percentile(y_true, q=i)
        booleaninterpercentile = tf.logical_and(tf.less(y_true,currpercentile),tf.greater(y_true,prevpercentile))
        trueslice = tf.boolean_mask(y_true, booleaninterpercentile)
        predslice = tf.boolean_mask(y_pred, booleaninterpercentile)
        sumofsquares += tf.to_float(K.square(K.mean(K.square(predslice - trueslice), axis=-1)))
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

    model.compile(optimizer='adam', loss=quadratic_mean_error, metrics=['mae'])
    return model

qmemodelrmae = collections.defaultdict(dict)
qmemodelrmse = collections.defaultdict(dict)
qmemodelurmae = collections.defaultdict(dict)
qmemodelurmse = collections.defaultdict(dict)
qmemodelrrmse = collections.defaultdict(dict)
qmemodelurrmse = collections.defaultdict(dict)

for stock1,stock2 in itertools.combinations(allcols,2):
    qmemodelrmae[stock1][stock2] = 0
    qmemodelrmse[stock1][stock2] = 0
    qmemodelurmae[stock1][stock2] = 0
    qmemodelurmse[stock1][stock2] = 0

for stock1,stock2 in itertools.permutations(allcols,2):
    print('stock1,stock2',stock1,stock2)
sys.exit()

for stock1,stock2 in itertools.permutations(allcols,2):
    print('stock1,stock2',stock1,stock2)
    Ypast, Ycurr = getstockdata(df[['Date', stock1]], p)
    # Ypast, Ycurr = getstockdata(df[['Date', 'MSFT_prices']])
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3*numrecords))
    numtrainrecords = int(math.ceil(0.7*numrecords))
    modelr = restricted_qme_model(p)
    np.random.seed(3)
    modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
    qme_mse_valuer, qme_mae_valuer = modelr.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
    print('\n')
    print('qme modelr Ycurrp.mean()',Ycurrp.mean())
    print('qme modelr Ycurrp.std()',Ycurrp.std())
    print('qme modelr mae_value',qme_mae_valuer)
    print('qme modelr mse_value',qme_mse_valuer)
    print('qme modelr r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
    qmemodelrmae[stock1][stock2] = qme_mae_valuer
    qmemodelrmse[stock1][stock2] = qme_mse_valuer
    qmemodelrrmse[stock1][stock2] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
    Ypast1, Ycurr1 = getstockdata(df[['Date', stock1]], p)
    Ypast2, Ycurr2 = getstockdata(df[['Date', stock2]], q)
    Ycurr2 = Ycurr1
    Ypast = np.concatenate((Ypast1, Ypast2))
    Ycurr = np.concatenate((Ycurr1, Ycurr2))
    # Ypast,Ycurr = getstockdata(df[['Date', stock2]])
    # Ypast,Ycurr = getstockdata(df[['Date','ORCL_prices']])
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3*numrecords))
    numtrainrecords = int(math.ceil(0.7*numrecords))
    modelur = restricted_qme_model(q)
    np.random.seed(7)
    modelur.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrp = modelur.predict(Ypast[-numtestrecords:], batch_size=128)
    qme_mse_valueur, qme_mae_valueur = modelur.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
    print('\n')
    print('qme modelur Ycurrp.mean()',Ycurrp.mean())
    print('qme modelur Ycurrp.std()',Ycurrp.std())
    print('qme modelur mae_value',qme_mae_valueur)
    print('qme modelur mse_value',qme_mse_valueur)
    print('qme modelur r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
    qmemodelurmae[stock1][stock2] = qme_mae_valueur
    qmemodelurmse[stock1][stock2] = qme_mse_valueur
    qmemodelurrmse[stock1][stock2] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
print('qmemodelrmae',qmemodelrmae)
print('qmemodelrmse',qmemodelrmse)
print('qmemodelurmae',qmemodelurmae)
print('qmemodelurmse',qmemodelurmse)
print('qmemodelrrmse', qmemodelrrmse)
print('qmemodelurrmse', qmemodelurrmse)
with open(QmemodelrmaePath, 'wb') as handle:
    pickle.dump(qmemodelrmae, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(QmemodelrmsePath, 'wb') as handle:
    pickle.dump(qmemodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(QmemodelurmaePath, 'wb') as handle:
    pickle.dump(qmemodelurmae, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(QmemodelurmsePath, 'wb') as handle:
    pickle.dump(qmemodelurmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(QmemodelrrmsePath, 'wb') as handle:
    pickle.dump(qmemodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(QmemodelurrmsePath, 'wb') as handle:
    pickle.dump(qmemodelurrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
msermses = {}
qmermses = {}
for stock in allcols[1:]:
    Ypast, Ycurr = getstockdata(df[['Date', stock]],p)
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3 * numrecords))
    numtrainrecords = int(math.ceil(0.7 * numrecords))
    modelmser = restricted_mse_model(p)
    np.random.seed(3)
    modelmser.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2,
               validation_split=0.1)
    Ycurrpmse = modelmser.predict(Ypast[-numtestrecords:], batch_size=128)
    msermses[stock] = math.sqrt(mean_squared_error(Ycurrpmse, Ycurr[-numtestrecords:]))
    modelqmer = restricted_qme_model(p)
    np.random.seed(3)
    modelqmer.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrpqme = modelqmer.predict(Ypast[-numtestrecords:], batch_size=128)
    qmermses[stock] = math.sqrt(mean_squared_error(Ycurrpqme, Ycurr[-numtestrecords:]))
print('mse modelr rmses', msermses)
print('qme modelr rmses', qmermses)
# ('mse modelr rmses', {'CAT_prices': 0.10638062186810399, 'APA_prices': 0.16925567933884197, 'SUN_prices': 0.20955881503342277, 'ABT_prices': 0.5020951712321089, 'LAKE_prices': 0.10566026298606304, 'UTX_prices': 0.43057081914404777, 'AEM_prices': 0.14326454813221084, 'T_prices': 0.033391623376135764, 'AAPL_prices': 0.7642329962727583, 'AFG_prices': 0.48511460987724503, 'MSFT_prices': 0.1897965685053637, 'WWD_prices': 0.31199758621355667, 'ORCL_prices': 0.49741264362718296, 'MCD_prices': 0.4258135092973906, 'IXIC_prices': 0.25665285776308516, 'B_prices': 0.28671578895086913})
# ('qme modelr rmses', {'CAT_prices': 0.5060790679800434, 'APA_prices': 0.24365732317159325, 'SUN_prices': 0.42714907145495856, 'ABT_prices': 0.6419927969369505, 'LAKE_prices': 0.16125048027785355, 'UTX_prices': 0.6280369733654089, 'AEM_prices': 0.23008945864521987, 'T_prices': 0.15762412743930138, 'AAPL_prices': 1.2210931139689403, 'AFG_prices': 0.8298099465850824, 'MSFT_prices': 0.3430728190329516, 'WWD_prices': 0.7443534501587998, 'ORCL_prices': 0.41429868389400526, 'MCD_prices': 0.652873317385026, 'IXIC_prices': 0.5612604507645994, 'B_prices': 0.7061986394913153})
msemodelrmae = collections.defaultdict(dict)
msemodelrmse = collections.defaultdict(dict)
msemodelrrmse = collections.defaultdict(dict)
msemodelurmae = collections.defaultdict(dict)
msemodelurmse = collections.defaultdict(dict)
msemodelurrmse = collections.defaultdict(dict)
for stock1,stock2 in itertools.permutations(allcols,2):
    print('stock1,stock2',stock1,stock2)
    Ypast, Ycurr = getstockdata(df[['Date', stock1]],p)
    # Ypast,Ycurr = getstockdata(df[['Date','MSFT_prices']])
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3*numrecords))
    numtrainrecords = int(math.ceil(0.7*numrecords))
    modelr = restricted_mse_model(p)
    np.random.seed(3)
    modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
    mse_mse_valuer, mse_mae_valuer = modelr.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
    print('\n')
    print('mse modelr Ycurrp.mean()',Ycurrp.mean())
    print('mse modelr Ycurrp.std()',Ycurrp.std())
    print('mse modelr mae_value',mse_mae_valuer)
    print('mse modelr mse_value',mse_mse_valuer)
    print('mse modelr r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
    msemodelrrmse[stock1][stock2] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
    msemodelrmae[stock1][stock2] = mse_mae_valuer
    msemodelrmse[stock1][stock2] = mse_mse_valuer
    Ypast1, Ycurr1 = getstockdata(df[['Date', stock1]],p)
    Ypast2, Ycurr2 = getstockdata(df[['Date', stock2]],q)
    Ycurr2 = Ycurr1
    Ypast = np.concatenate((Ypast1, Ypast2))
    Ycurr = np.concatenate((Ycurr1, Ycurr2))
    # Ypast, Ycurr = getstockdata(df[['Date', stock2]])
    # Ypast,Ycurr = getstockdata(df[['Date','ORCL_prices']])
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3*numrecords))
    numtrainrecords = int(math.ceil(0.7*numrecords))
    modelur = restricted_mse_model(q)
    np.random.seed(7)
    modelur.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrp = modelur.predict(Ypast[-numtestrecords:], batch_size=128)
    mse_mse_valueur, mse_mae_valueur = modelur.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
    print('\n')
    print('mse modelur Ycurrp.mean()',Ycurrp.mean())
    print('mse modelur Ycurrp.std()',Ycurrp.std())
    print('mse modelur mae_value',mse_mae_valueur)
    print('mse modelur mse_value',mse_mse_valueur)
    print('mse modelur r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
    msemodelurrmse[stock1][stock2] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
    msemodelurmae[stock1][stock2] = mse_mae_valueur
    msemodelurmse[stock1][stock2] = mse_mse_valueur
print('msemodelrmae', msemodelrmae)
print('msemodelrmse', msemodelrmse)
print('msemodelrrmse', msemodelrrmse)
print('msemodelurmae', msemodelurmae)
print('msemodelurmse', msemodelurmse)
print('msemodelurrmse', msemodelurrmse)
with open(MsemodelrmaePath, 'wb') as handle:
    pickle.dump(msemodelrmae, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MsemodelrmsePath, 'wb') as handle:
    pickle.dump(msemodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MsemodelrrmsePath, 'wb') as handle:
    pickle.dump(msemodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MsemodelurmaePath, 'wb') as handle:
    pickle.dump(msemodelurmae, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MsemodelurmsePath, 'wb') as handle:
    pickle.dump(msemodelurmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(MsemodelurrmsePath, 'wb') as handle:
    pickle.dump(msemodelurrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
