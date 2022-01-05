import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Frac Diff FFD
def get_weight_ffd(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(x, d, thres=1e-5):
    w = get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width:i + 1])[0])
    return np.array(output)


# plot ffd weights
def getWeights(d,lags):
    # get weights for plot, no threshold
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1,1) 
    return w

def plotWeights(dRange, lags, numberPlots):
    weights = pd.DataFrame(np.zeros((lags, numberPlots)))
    interval = np.linspace(dRange[0], dRange[1], numberPlots)
    
    for i, diff_order in enumerate(interval):
        weights[i] = getWeights(diff_order,lags)
    weights.columns = [round(x,2) for x in interval]
    
    fig = weights.plot(figsize=(15,6))
    plt.legend(title='Order of differencing', loc='upper right')
    plt.title('Lag coefficients for various orders of differencing')
    plt.xlabel('lag coefficients')
    #plt.grid(False)
    plt.show()


    plt.show()
