import numpy as np
from scipy import interpolate as interp
from scipy.fftpack import dct
from numpy import linalg as LA
import scipy.optimize as opt
import matplotlib.pyplot as plt


def smoothn(yin, w=None, s=None, robust=True, tolZ=1.0e-5, maxIter=100):
    """Perfom the penalized least-squares smoothing of data of Garcia, D. (2010)
       http://www.biomecardio.com/matlab/smoothn.html
       The smoothing allows for iterative robust smoothing with missing data.
       The smoothing parameter can be automatically determined using a 
       generalized cross-validation score method.
       Originally implemented in MATLAB by
       AUTHOR Damien Garcia
       Ported to python by
       AUTHOR: Christopher J Burke
       ***Currently limited to the 1D case***
       For missing, corrupted, or in-transit data that you dont want
       to influence the fit it, is not sufficient to set the weight
       value to 0 for the bad data points.  In addition to setting
       the weight=0, you MUST also do one of the two following choices.
       1) Also set the bad data to NaN in the data vector (y), and this
          function will use linear interpolation across the gap
          to fill in approximate values OR
       2) Interpolate across the gap to fill in the bad data in the data vector
          before calling this function
       INPUT:
       yin - data vector one wants to find a smoothing function for
       w - [0-1] data weights
       s - smoothing parameter if not specified it is determined with GCVS
       robust - Perform iterative reweighting for outliers.
       tolZ - relative tolerance for change between iterations
       maxIter - maximum number of iterations for convergence
       OUTPUT:
       z - smoothed model for data vector
       w - final weighting vector
       s - final smoothing parameter
       exitflag - flag if solution converged before maxIter
    """
    
    # Force y to be numpy double array and a copy
    y = np.array(yin, dtype=np.double, copy=True)
    sizy = y.size
    noe = sizy
    if noe < 2: # Too few elements return and do nothging
        z = y
        return z
    # Check for weights
    # weighted fit is performed if w vector is an argument OR 
    # non finite values appear in data vector    
    isWeighted = False
    if w is None:
        w = np.full_like(y, 1.0)
    else:
        isWeighted = True
    isFinite = np.isfinite(y)
    nof = isFinite.sum()
    if not isFinite.all():    
        isWeighted = True
    
    w = np.where(isFinite, w, 0.0)
    w = w / w.max()
    # autosmoothing
    isAuto = False
    if s is None:
        isAuto = True
    
    # Creation of the Lambda tensor
    lam = np.zeros_like(y)
    lam = -2.0 + 2.0 * np.cos((np.linspace(1.0,sizy,sizy)-1.0)*np.pi/sizy)
    
    if not isAuto:
        gamma = 1.0 / (1.0 + s * lam**2)
    
    #Upper and lower bounds of smoothness parameter
    hMin = 5.0e-3
    hMax = 0.99
    usePow = 2.0 
    tmp = 1.0 + np.sqrt(1.0 + 8.0 * hMax**usePow)
    sMinBnd = ((tmp / 4.0 / hMax**usePow)**2 - 1.0) / 16.0
    tmp = 1.0 + np.sqrt(1.0 + 8.0 * hMin**usePow)
    sMaxBnd = ((tmp / 4.0 / hMin**usePow)**2 - 1.0) / 16.0
    
    # Initialize a rough guess at the smooth function if weighting is involved
    wTot = w
    if isWeighted:
        z = initialGuess(y, np.isfinite(y))
    else:
        z = np.zeros_like(y)
    z0 = z
    # Do linear interpolation for nans in data vector    
    if not isFinite.all():
        fullx = np.arange(len(y))
        gdIdx = np.where(isFinite)[0]
        tmpx = fullx[gdIdx]
        tmpy = y[gdIdx]
        funcinterp = interp.interp1d(tmpx, tmpy, kind='linear')
        y = funcinterp(fullx)
    tol = 1.0
    robustIterativeProcess = True
    robustStep = 1
    nit = 0
    # Relaxation Factor
    RF = 1.0
    if isWeighted:
        RF = RF + 0.75
    
    # Main iterative Loop
    while robustIterativeProcess:
        # amount of weights
        aow = wTot.sum() / noe
        while tol > tolZ and nit < maxIter:
            nit = nit + 1
            dcty = dct(wTot * (y - z) + z, type=2, norm='ortho')
            if isAuto and np.remainder(np.log2(nit),1) == 0:
                allOutput = opt.minimize_scalar(gcv, \
                        bounds=[np.log10(sMinBnd),np.log10(sMaxBnd)], \
                        args=(y, lam, dcty, wTot, isFinite, aow, noe, nof), \
                                 method='bounded', tol=None, \
                                 options={'xatol':1.0e-1})
                p = allOutput['x']
                s = 10.0**p
                gamma = 1.0 / (1.0 + s * lam**2)
            z = RF * dct(gamma * dcty, type=3, norm='ortho') + (1.0 - RF) * z
            tol = LA.norm(z0 - z) / LA.norm(z)
            if not isWeighted: # if no weighted/missing data tol=0.0 (no iter)
                tol = 0.0
            z0 = z # save last output
        exitFlag = nit < maxIter
        
        if robust: # robust smoothing iteratively re-weight outliers
            # average leverage
            h = np.sqrt(1.0 + 16.0 * s)
            h = np.sqrt(1.0 + h) / np.sqrt(2.0) / h
            # take robust weights into account
            wTot = w * robustWeights(y-z, isFinite, h)
            # reinitialize for another iteration
            isWeighted = True
            tol = 1.0
            nit = 0
            robustStep = robustStep +1
            robustIterativeProcess = robustStep < 4 # Max of 3 robust steps
        else:
            robustIterativeProcess = False # No iterations needed
    return z, w, s, exitFlag
    
def initialGuess(y, iFin ):
    z = y
    if not iFin.all():
        # Do linear interpolation for missing NaN data
        fullx = np.arange(len(y))
        gdIdx = np.where(iFin)[0]
        tmpx = fullx[gdIdx]
        tmpy = y[gdIdx]
        funcinterp = interp.interp1d(tmpx, tmpy, kind='linear')
        z = funcinterp(fullx)
    z = dct(z, type=2, norm='ortho')
    zeroIdx = np.ceil(len(z)/10)
    z[zeroIdx:] = 0.0
    z = dct(z, type=3, norm='ortho')
    return z
    
def gcv(p, y, lam, dcty, wTot, iFin, aow, noe, nof):
    s = 10.0**p
    gamma = 1.0 / (1.0 + s * lam**2)
    if aow > 0.9:
        rss = LA.norm(dcty * (gamma - 1.0))**2
    else:
        yhat = dct(gamma * dcty, type=3, norm='ortho')
        gdIdx = np.where(iFin)[0]
        rss = LA.norm(np.sqrt(wTot[gdIdx]) * (y[gdIdx] - 
              yhat[gdIdx]))**2
    trH = gamma.sum()
    return rss / nof / (1.0 - trH/noe)**2

def robustWeights(r, iFin, h):
    gdIdx = np.where(iFin)[0]
    mad = np.median(abs(r[gdIdx] - np.median(r[gdIdx]))) #median abs deviation
    u = np.abs(r / (1.4826 * mad) / np.sqrt(1.-h)) # studentized residuals
    c = 4.685
    u = u / c
    u2 = u * u
    w = (1.0 - u2)**2
    w = np.where(u > 1.0, 0.0, w)
    w = np.where(np.logical_not(iFin), 0.0, w)
    w = np.where(np.logical_not(np.isfinite(w)), 0.0, w)
    return w
    
# Run the test of the smoothn
if __name__ == "__main__":
    x = np.linspace(0,100,2**8)
    y = np.cos(x/10.0)+(x/50.0)**2 + np.random.randn(len(x))/10.0;
    y[[69, 74, 79]] = np.array([5.5, 5, 6])
    plt.plot(x,y,'.')
    z = smoothn(y, robust=False) # Regular smoothing
    plt.plot(x,z[0],'-r')    
    plt.show()
    zr = smoothn(y, robust=True) # Robust smoothing
    plt.plot(x,y,'.')
    plt.plot(x,zr[0],'-r')    
    plt.show()
    ynew = np.array(y, copy=True)
    ynew[100:110] = np.nan
    zmr = smoothn(ynew, robust=True)
    plt.plot(x,ynew,'.')
    plt.plot(x,zmr[0],'-r')    
    plt.show()
    
   
       