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

import networkx as nx

import collections
import pickle
from sklearn.metrics import mean_squared_error

# get lagged data
def getlagdata(dfone, lag, inputbatchsize):
    Ypast = []
    Ycurr = []
    for i in range(-inputbatchsize, 0):
        #y = dfone.iloc[i,1]
        #x = dfone.iloc[i - lag:i,1].tolist()
        y = dfone.iloc[i]
        x = dfone.iloc[i - lag:i].tolist()
        Ypast.append(x)
        Ycurr.append(y)
    Ypast = np.vstack(Ypast)
    Ycurr = np.vstack(Ycurr)
    Ycurr = Ycurr.reshape(Ycurr.shape[0], )
    return Ypast,Ycurr

# deep learning model for autoregression
def regression_model(lag, ypast_dim1):
    model = Sequential()
    model.add(Dense(units=2*lag, activation='relu', kernel_initializer='normal', bias_initializer='zeros', input_dim=ypast_dim1))
    model.add(Dropout(0.5))
    model.add(Dense(int(lag/2), activation='linear', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(int(lag/2), activation='relu', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='normal', bias_initializer='zeros'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.compile(loss='mse', optimizer='adam', metrics=['msd'])
    return model

# fn to calculate f statistic
def fstat(rmse_ur, rmse_r):
    return (rmse_r-rmse_ur)/rmse_ur

# fn to construct granger features graph from bivariate regression
def construct_graph(rmse_ur, rmse_r):
    rmse_keys = list(rmse_ur.keys())
    #graph_dict = collections.defaultdict(dict)
    graph_dict = {k:[] for k in rmse_ur}
    print(graph_dict)
    for k in rmse_keys:
        cand = rmse_ur[k]
        #graph_dict[k] = []
        for c in cand:
            fs = fstat(rmse_ur[k][c], rmse_r[k][c])
            if fs > 0.05:
                graph_dict[c].append(k)
        #if graph_dict[c] == []:
        #    graph_dict.pop(k, None)
    graph_dict = {k:v for (k,v) in graph_dict.items() if v != []}
    G = nx.DiGraph(graph_dict, directed=True)
    return G, graph_dict

def run_main(args):
    """
    df = pd.read_csv("pricesvolumes.csv")
    cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]
    #cols = [1,2,3,4]

    df.drop(df.columns[cols],axis=1,inplace=True)
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
    """
    
    df = args['alg_loader'].dataset.data
    allcols = df.columns.tolist()
    print(df)
    batch_s = args['batch_size']
    inputbatchsize = args['inputbatchsize']
    # lag variables for r/ur models
    p = args['p']
    q = args['q']
    numepochs = args['train_epochs']

    model_r_mae = collections.defaultdict(dict)
    model_r_mse = collections.defaultdict(dict)
    model_r_rmse = collections.defaultdict(dict)
    model_ur_mae = collections.defaultdict(dict)
    model_ur_mse = collections.defaultdict(dict)
    model_ur_rmse = collections.defaultdict(dict)

    for stock1,stock2 in itertools.permutations(allcols,2):
        print('stock1,stock2',stock1,stock2)
        #Ypast, Ycurr = getlagdata(df[['Date', stock1]],p,inputbatchsize)
        Ypast, Ycurr = getlagdata(df[stock1],p,inputbatchsize)
        # Ypast,Ycurr = getlagdata(df[['Date','MSFT_prices']])
        numrecords = len(Ycurr)
        numtestrecords = int(math.ceil(0.3*numrecords))
        numtrainrecords = int(math.ceil(0.7*numrecords))
        model_r = regression_model(p, Ypast.shape[1])
        np.random.seed(3)
        model_r.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=batch_s, verbose=2, validation_split=0.1)
        Ycurrp = model_r.predict(Ypast[-numtestrecords:], batch_size=128)
        mse_mse_value_r, mse_mae_value_r = model_r.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
        print('\n')
        print('mse modelr Ycurrp.mean()',Ycurrp.mean())
        print('mse modelr Ycurrp.std()',Ycurrp.std())
        print('mse modelr mae_value',mse_mae_value_r)
        print('mse modelr mse_value',mse_mse_value_r)
        print('mse modelr r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
        model_r_rmse[stock1][stock2] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        model_r_mae[stock1][stock2] = mse_mae_value_r
        model_r_mse[stock1][stock2] = mse_mse_value_r

        #Ypast1, Ycurr1 = getlagdata(df[['Date', stock1]],p,inputbatchsize)
        #Ypast2, Ycurr2 = getlagdata(df[['Date', stock2]],q,inputbatchsize)
        Ypast1, Ycurr1 = getlagdata(df[stock1],p,inputbatchsize)
        Ypast2, Ycurr2 = getlagdata(df[stock2],q,inputbatchsize)
        Ycurr2 = Ycurr1
        Ypast = np.concatenate((Ypast1, Ypast2))
        Ycurr = np.concatenate((Ycurr1, Ycurr2))

        numrecords = len(Ycurr)
        numtestrecords = int(math.ceil(0.3*numrecords))
        numtrainrecords = int(math.ceil(0.7*numrecords))
        model_ur = regression_model(q, Ypast.shape[1])
        np.random.seed(7)
        model_ur.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=batch_s, verbose=2, validation_split=0.1)
        Ycurrp = model_ur.predict(Ypast[-numtestrecords:], batch_size=128)
        mse_mse_value_ur, mse_mae_value_ur = model_ur.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
        print('\n')
        print('mse modelur Ycurrp.mean()',Ycurrp.mean())
        print('mse modelur Ycurrp.std()',Ycurrp.std())
        print('mse modelur mae_value',mse_mae_value_ur)
        print('mse modelur mse_value',mse_mse_value_ur)
        print('mse modelur r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
        model_ur_rmse[stock1][stock2] = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))
        model_ur_mae[stock1][stock2] = mse_mae_value_ur
        model_ur_mse[stock1][stock2] = mse_mse_value_ur

    print('msemodelrmae', model_r_mae)
    print('msemodelrmse', model_r_mse)
    print('msemodelrrmse', model_r_rmse)
    print('msemodelurmae', model_ur_mae)
    print('msemodelurmse', model_ur_mse)
    print('msemodelurrmse', model_ur_rmse)

    # with open(MsemodelrmaePath, 'wb') as handle:
    #     pickle.dump(model_r_mae, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(MsemodelrmsePath, 'wb') as handle:
    #     pickle.dump(model_r_mse, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(MsemodelrrmsePath, 'wb') as handle:
    #     pickle.dump(model_r_rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(MsemodelurmaePath, 'wb') as handle:
    #     pickle.dump(model_ur_mae, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(MsemodelurmsePath, 'wb') as handle:
    #     pickle.dump(model_ur_mse, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(MsemodelurrmsePath, 'wb') as handle:
    #     pickle.dump(model_ur_rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)


#if __name__ == '__main__':
#    run_main(args)