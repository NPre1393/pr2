import numpy as np
import pandas as pd
from scipy.sparse import random
import random as rd
import pandas as pd
import matplotlib.pyplot as plt

class dataset():

    def __init__(self, features=10, lag=2, dep_dens=0.3, dependencies1=0, dependencies2=0, n1=200, n2=100, dists=0, coeff_min=0.6, coeff_max=0.9, data=0, caused_ts=0):
        self.features = features
        self.lag = lag
        self.dep_dens = dep_dens
        if not caused_ts:
            self.caused_ts = np.random.randint(self.features/2)+1
        else:
            self.caused_t = caused_ts

        if not dependencies1 == 0:
            self.dependencies1 = dependencies1
        else:
            self.dependencies1 = self.gen_dependencies(features, dep_dens, self.caused_ts)

        if not dependencies2 == 0:
            self.dependencies2 = dependencies2
        else:
            self.dependencies2 = self.gen_dependencies(features, dep_dens, self.caused_ts)
        
        self.n1 = n1
        self.n2 = n2

        if not dists:
            self.dists = np.zeros(self.features)
        else:
            self.dists = dists
        self.coeff_min = coeff_min
        self.coeff_max = coeff_max
        self.data = data
        self.coefficients1 = 0
        self.coefficients2 = 0
        self.GC_output = 0
    
    def __repr__(self):
        repr = 'Dataset Information\n'\
            'Features = {}, Lag = {}\n'\
            'dists = {}\n'\
            'dep1:\n {}\n'\
            'dep2:\n {}'.format(self.features, self.lag, self.dists, self.dependencies1, self.dependencies2)

        return repr

    ###
    # Function to generate dependency structure of time series
    # features: number of features in data set
    # density: approx. nr of ts influenced by causal_drivers
    # caused_ts: nr of ts generated by other ts
    ###
    def gen_dependencies(self, features, density, caused_ts=3):
        dep_structure = np.zeros([features,features])

        for i in range(caused_ts):
            dep = random(1,features-caused_ts,density=density,format='csr')
            dep.data[:] = 1
            dep_structure[i,caused_ts:] = dep.A
        """
        dep = random(features,features,density=density,format='csr')
        dep.data[:] = 1
        dep_structure = dep.A
        dep_structure[0:caused_ts,0:caused_ts] = 0
        dep_structure[caused_ts:] = 0
        for i in range(caused_ts):
            if not np.nonzero(dep_structure[i])[0].size:
        """
        #print(dep_structure)
        return dep_structure

    ###
    # Function to generate data using dependency structures to simulate a change in
    # causal dependency
    # dependencies1 determine 0:n1 values
    # dependencies2 determine n1:n1+n2 values
    ###
    def gen_dep_anom_data(self):
        full_len = self.n1+self.n2
        data = np.zeros([self.features, full_len])
        # row indcies of caused ts from dependency structure 1 and 2
        rowIdx1, colIdx1 = np.where(self.dependencies1 == 1)
        deps1 = np.unique(rowIdx1)
        rowIdx2, colIdx2 = np.where(self.dependencies2 == 1)
        deps2 = np.unique(rowIdx2)
        deps1_len = len(deps1)
        deps2_len = len(deps2)
        # number of dependencies per caused ts
        deps1_per = np.sum(self.dependencies1, axis=1).astype(int)
        deps2_per = np.sum(self.dependencies2, axis=1).astype(int)

        # generate time series that do not have any dependencies first -> they will generate the
        # caused time series
        for i in range(self.features):
            if i not in deps1:
                if self.dists[i] == 0:
                    mu = np.random.random_sample()
                    sigma = np.random.random_sample()
                    data[i] = np.random.normal(mu, sigma, full_len)
                elif self.dists[i] == 1:
                    data[i] = np.random.poisson(1,full_len)
                elif self.dists[i] == 2:
                    data[i] = np.random.binomial(1,0.5,full_len)
                elif self.dists[i] == 3:
                    data[i] = np.random.gamma(1,1,full_len)
                else:
                    return "dataset dists vector contains wrong values, only 0-3 allowed" 

        # generate each sample by 
        for i in range(deps1_len):
            mu = np.random.random_sample()
            sigma = np.random.random_sample()            
            #coeffs = np.array([round(rd.uniform(self.coeff_min, self.coeff_max),1) for _ in range(deps1_per[i]*self.lag)])
            coeffs = np.array([round(x,1) for x in np.random.uniform(self.coeff_min, self.coeff_max, deps1_per[i]*self.lag)])
            for j in range(self.lag,self.n1):
                #if j < self.lag+1:
                #    data[i,j] = np.random.normal(mu,sigma)
                #if j > self.lag+1:
                    res = (data[:,j-self.lag:j-1]).T*self.dependencies1[deps1[i]]
                    lagged_vals = np.zeros(coeffs.shape)
                    lagged_vals[:len(res[np.nonzero(res)])] = res[np.nonzero(res)]
                    data[i,j] = sum(coeffs*lagged_vals)
            # linear function to impute 0:lag values
            A = np.vstack([[x for x in range(self.n1-self.lag)], np.ones(self.n1-self.lag)]).T
            k, d = np.linalg.lstsq(A, data[i,self.lag:self.n1], rcond=None)[0]
            data[i,:self.lag] = [k*x+d for x in range(self.lag)]

        for i in range(deps2_len):
            coeffs = np.array([round(x,1) for x in np.random.uniform(self.coeff_min, self.coeff_max, deps2_per[i]*self.lag)])            
            for j in range(self.n1,full_len):
                res = (data[:,j-self.lag:j-1]).T*self.dependencies2[deps2[i]]
                lagged_vals = np.zeros(coeffs.shape)
                lagged_vals[:len(res[np.nonzero(res)])] = res[np.nonzero(res)]
                data[i,j] = sum(coeffs*lagged_vals)
                
        self.data = pd.DataFrame(data=data.T,index=range(full_len),columns=range(self.features))
                
    def plot_input(self):
        fig, ax = plt.subplots(1,2,figsize=(16,5))
        # plot all data
        ax[0].plot(self.data)
        # plot first 100 time steps
        ax[1].plot(self.data[:100])
        plt.show()

    def plot_output_anom(self):
        pass

    def plot_output_GC(self, GC_est):
        print('True variable usage = %.2f%%' % (100 * np.mean(self.GC)))
        print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
        print('Accuracy = %.2f%%' % (100 * np.mean(self.GC == GC_est)))

        # Make figures
        fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
        axarr[0].imshow(self.GC, cmap='Blues')
        axarr[0].set_title('GC actual')
        axarr[0].set_ylabel('Affected series')
        axarr[0].set_xlabel('Causal series')
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])

        axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, p, p, 0))
        axarr[1].set_title('GC estimated')
        axarr[1].set_ylabel('Affected series')
        axarr[1].set_xlabel('Causal series')
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])

        # Mark disagreements
        for i in range(self.features):
            for j in range(self.features):
                if self.GC[i, j] != GC_est[i, j]:
                    rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
                    axarr[1].add_patch(rect)

        plt.show()



    def make_var_stationary(self, beta, radius=0.97):
        '''Rescale coefficients of VAR model to make stable.'''
        p = beta.shape[0]
        lag = beta.shape[1] // p
        bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
        beta_tilde = np.vstack((beta, bottom))
        eigvals = np.linalg.eigvals(beta_tilde)
        max_eig = max(np.abs(eigvals))
        nonstationary = max_eig > radius
        if nonstationary:
            return self.make_var_stationary(0.95 * beta, radius)
        else:
            return beta

    def simulate_var(self, p, T, lag, sparsity=0.3, beta_value=1.0, sd=0.1, seed=0):
        p = self.features
        lag = self.lag
        T = self.n1 + self.n2
        sparsity = self.dep_dens

        if seed is not None:
            np.random.seed(seed)

        # Set up coefficients and Granger causality ground truth.
        GC = np.eye(p, dtype=int)
        beta = np.eye(p) * beta_value

        num_nonzero = int(p * sparsity) - 1
        for i in range(p):
            choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
            choice[choice >= i] += 1
            beta[i, choice] = beta_value
            GC[i, choice] = 1

        beta = np.hstack([beta for _ in range(lag)])
        beta = self.make_var_stationary(beta)

        # Generate data.
        burn_in = 100
        errors = np.random.normal(scale=sd, size=(p, T + burn_in))
        X = np.zeros((p, T + burn_in))
        X[:, :lag] = errors[:, :lag]
        for t in range(lag, T + burn_in):
            X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
            X[:, t] += + errors[:, t-1]
        
        self.data = X.T[burn_in:]
        self.dependencies1 = GC
        self.beta = beta
        #return X.T[burn_in:], beta, GC
        

    def lorenz(self, x, t, F):
        '''Partial derivatives for Lorenz-96 ODE.'''
        p = len(x)
        dxdt = np.zeros(p)
        for i in range(p):
            dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

        return dxdt


    def simulate_lorenz_96(self, p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                        seed=0):
        if seed is not None:
            np.random.seed(seed)

        p = self.features
        T = self.n1 + self.n2

        # Use scipy to solve ODE.
        x0 = np.random.normal(scale=0.01, size=p)
        t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
        X = odeint(self.lorenz, x0, t, args=(F,))
        X += np.random.normal(scale=sd, size=(T + burn_in, p))

        # Set up Granger causality ground truth.
        GC = np.zeros((p, p), dtype=int)
        for i in range(p):
            GC[i, i] = 1
            GC[i, (i + 1) % p] = 1
            GC[i, (i - 1) % p] = 1
            GC[i, (i - 2) % p] = 1

        #return X[burn_in:], GC
        self.data = X[burn_in:]
        self.dependencies1 = GC
