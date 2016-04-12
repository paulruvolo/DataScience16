from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np

dist = multivariate_normal([0, 0], [[1, 0],[0, 1]])
poverty = np.random.randn(100,1)
samples = dist.rvs(100) + np.hstack((poverty, -poverty))
data = pd.DataFrame({'EMF_exposure': samples[:,0],
					 'health': samples[:,1],
					 'poverty': poverty.ravel()})