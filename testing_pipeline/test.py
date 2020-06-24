import data_generator as dg
import pandas as pd
import matplotlib.pyplot as plt

for i in range(1000):
    ds = dg.dataset()
    print(ds)
    ds.gen_data()
    
ds = dg.dataset(features=20,n1=1000,n2=500)
ds.gen_data()
data = ds.data

print(data[:5]) 
print(data.iloc[:,0:5])
print(data.loc[:,0:5])
print(data.loc[:,[1,2,3]])
print(data[2])