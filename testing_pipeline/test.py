import data_generator as dg
import pandas as pd
import matplotlib.pyplot as plt

ds = dg.dataset(features=10,n1=200,n2=100)
ds.gen_dep_anom_data()
data = ds.data

print(data[:5]) 
print(data.iloc[:,0:5])
print(data.loc[:,0:5])
print(data.loc[:,[1,2,3]])
print(data[2])

ds.plot_input()

