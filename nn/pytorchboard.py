import matplotlib.pyplot as plt
import numpy as np
import os 
from sklearn import linear_model

run_name = '21Jul18_151753_losses'

cwd = os.getcwd()
try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project")
    cwd = os.getcwd()
except: # for cluster run
    cwd = os.getcwd()
checkpoint_dir = cwd + '/nn/runs/'

losses = np.loadtxt(checkpoint_dir+run_name+'.csv')

regr = linear_model.LinearRegression()
x = np.linspace(1,len(losses),len(losses))  
regr.fit(np.reshape(x[200:], (-1,1)), losses[200:])
xnew = np.linspace(1,len(losses),4) 
ynew = regr.predict(np.reshape(xnew, (-1,1)))
plt.plot(x[200:], losses[200:], xnew,ynew,'-')
plt.show()