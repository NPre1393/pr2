import sys, os
# adds the parent directory path + apps folder appended to system path
#sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'apps'))
# adds the parent directory (gc_testing) to system path, all imports are relative to this dir 
sys.path.append(os.path.dirname(sys.path[0]))

import apps.data_generator as dg
import pandas as pd
import matplotlib.pyplot as plt

# generate 3 dataset objects with different nr of features and length

# generate available datasets 
# dataset that simulates dependency anomalies with a VAR model and 
# changing dependencies after n1 time steps (two dependency structures)
ds = dg.dataset(features=10,n=300,lag=3,dep_dens=0.6)
ds.gen_var_data()

#ds.plot_input()
ds.GC = ds.dependencies['dep1']

#ds.plot_causeEffect(effect=4, causes=[1,2,3,5])
ds.plot_causeEffect(effect=4)