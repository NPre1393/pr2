import data_generator as dg
import pandas as pd
import matplotlib.pyplot as plt

# generate 3 dataset objects with different nr of features and length
ds = dg.dataset(features=10,n=300,lag=3,dep_dens=0.6)
ds2 = dg.dataset(features=10,n=600)
ds3 = dg.dataset(features=15,n=1500)

# generate available datasets 
# dataset that simulates dependency anomalies with a VAR model and 
# changing dependencies after n1 time steps (two dependency structures)
ds.gen_dep_anom_data(n1=200,n2=100)
# dataset that simulates linear causal relations with a VAR model
ds2.gen_var_data()
# dataset that simulates dynamic causal relations with the lorenz96 ODE model
ds3.gen_lorenz96_data()

print(ds)
print(ds2)
print(ds3)

ds.plot_input()
#ds2.plot_input()
#ds3.plot_input()

# data is accessible from dataset object as numpy arrays saved as data variable
data = ds.data

print(data[:5]) 
print(data.iloc[:,0:5])
print(data.loc[:,0:5])
print(data.loc[:,[1,2,3]])
print(data[2])

#ds.plot_input()
ds.GC = ds.dependencies
#ds.plot_output_GC(ds.dependencies2)