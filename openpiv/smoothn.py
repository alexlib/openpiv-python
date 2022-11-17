""" Robust Smoothing """
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

def peaks(n):
    """
  Mimic basic of matlab peaks fn
  """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in range(n // 5):
        x0 = np.random.random() * n
        y0 = np.random.random() * n
        sdx = np.random.random() * n / 4.0
        sdy = sdx
        c = np.random.random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - (((x - x0) / sdx)) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= np.random.random()
        z += f
    return z


def test1():
    plt.figure()
    # 1-D example
    x = np.linspace(0, 100, 2 ** 8)
    y = np.cos(x / 10) + (x / 50) ** 2 + np.random.randn(x.size) / 10
    y[[70, 75, 80]] = [5.5, 5, 6]
    z, _,_,_ = smoothn(y)
    # Regular smoothing
    zr, _,_,_ = smoothn(y, robust=True)
    # Robust smoothing
    plt.subplot(121)
    plt.plot(x, y, "r.")
    plt.plot(x, z, "k")
    plt.title("Regular smoothing")
    plt.subplot(122)
    plt.plot(x, y, "r.")
    plt.plot(x, zr, "k")
    plt.title("Robust smoothing")
    plt.show()


def test2(axis=None):
    # 2-D example
    plt.figure(2)
    plt.clf()
    xp = np.arange(0, 1, 0.02)
    [x, y] = np.meshgrid(xp, xp)
    f = np.exp(x + y) + np.sin((x - 2 * y) * 3)
    fn = f + (np.random.randn(f.size) * 0.5).reshape(f.shape)
    fs = smoothn(fn)[0]
    plt.subplot(121)
    plt.imshow(fn, interpolation="Nearest")
    # axis square
    plt.subplot(122)
    plt.imshow(fs, interpolation="Nearest")
    # axis 
    plt.show()


def test3(axis=None):
    # 2-D example with missing data
    plt.figure(3)
    plt.clf()
    n = 256
    y0 = peaks(n)
    y = (y0 + np.random.random(y0.shape[0]) * 2 - 1.0).flatten()
    I = np.random.permutation(range(n ** 2))
    y[I[1: n ** 2 * 0.5]] = np.nan
    # lose 50% of data
    y = y.reshape(y0.shape)
    y[40:90, 140:190] = np.nan
    # create a hole
    yData = y.copy()
    z0 = smoothn(yData)
    # smooth data
    yData = y.copy()
    z = smoothn(yData, robust=True)
    # smooth data
    y = yData
    vmin = np.min([np.min(z), np.min(z0), np.min(y), np.min(y0)])
    vmax = np.max([np.max(z), np.max(z0), np.max(y), np.max(y0)])
    plt.subplot(221)
    plt.imshow(y, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("Noisy corrupt data")
    plt.subplot(222)
    plt.imshow(z0, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("Recovered data #1")
    plt.subplot(223)
    plt.imshow(z, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("Recovered data #2")
    plt.subplot(224)
    plt.imshow(y0, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("... compared with original data")


def test4(i=10, step=0.2, axis=None):
    [x, y, z] = np.mgrid[-2:2:step, -2:2:step, -2:2:step]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xslice = [-0.8, 1]
    yslice = 2
    zslice = [-2, 0]
    v0 = x * np.exp(-(x ** 2) - y ** 2 - z ** 2)
    vn = v0 + np.random.randn(x.size).reshape(x.shape) * 0.06
    v = smoothn(vn)[0]
    plt.figure(4)
    plt.clf()
    vmin = np.min([np.min(v[:, :, i]), np.min(
        v0[:, :, i]), np.min(vn[:, :, i])])
    vmax = np.max([np.max(v[:, :, i]), np.max(
        v0[:, :, i]), np.max(vn[:, :, i])])
    plt.subplot(221)
    plt.imshow(v0[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("clean z=%d" % i)
    plt.subplot(223)
    plt.imshow(vn[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("noisy")
    plt.subplot(224)
    plt.imshow(v[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("cleaned")


def test5():
    t = np.linspace(0, 2 * np.pi, 1000)
    x = 2 * np.cos(t) * (1 - np.cos(t)) + np.random.randn(t.shape[0]) * 0.1
    y = 2 * np.sin(t) * (1 - np.cos(t)) + np.random.randn(t.shape[0]) * 0.1
    zx = smoothn(x)
    zy = smoothn(y)
    plt.figure(5)
    plt.clf()
    plt.title("Cardioid")
    plt.plot(x, y, "r.")
    plt.plot(zx, zy, "k")


def test6(noise=0.05, nout=30):
    plt.figure(6)
    plt.clf()
    [x, y] = np.meshgrid(np.linspace(0, 1, 24), np.linspace(0, 1, 24))
    Vx0 = np.cos(2 * np.pi * x + np.pi / 2) * np.cos(2 * np.pi * y)
    Vy0 = np.sin(2 * np.pi * x + np.pi / 2) * np.sin(2 * np.pi * y)
    Vx = Vx0 + noise * np.random.randn(24, 24)
    # adding Gaussian noise
    Vy = Vy0 + noise * np.random.randn(24, 24)
    # adding Gaussian noise
    I = np.random.permutation(range(Vx.size))
    Vx = Vx.flatten()
    Vx[I[:nout]] = (np.random.rand(nout) - 0.5) * 5
    # adding outliers
    Vx = Vx.reshape(Vy.shape)
    Vy = Vy.flatten()
    Vy[I[0:nout]] = (np.random.rand(nout) - 0.5) * 5
    # adding outliers
    Vy = Vy.reshape(Vx.shape)
    Vsx = smoothn(Vx)
    Vsy = smoothn(Vy)
    plt.subplot(131)
    plt.quiver(x, y, Vx, Vy, 2.5)
    plt.title("Noisy")
    plt.subplot(132)
    plt.quiver(x, y, Vsx, Vsy)
    plt.title("Recovered")
    plt.subplot(133)
    plt.quiver(x, y, Vx0, Vy0)
    plt.title("Original")


def sparseSVD(D):
    import scipy.sparse
    Ds = scipy.sparse.csc_matrix(D)
    a = scipy.sparse.linalg.svds(Ds, Ds.shape[0])
    return a


def sparseTest(n=1000):
    I = np.identity(n)

    # define a 'traditional' D1 matrix
    # which is a right-side difference
    # and which is *not* symmetric :-(
    D1 = np.matrix(I - np.roll(I, 1))
    # so define a symemtric version
    D1a = D1.T - D1

    U, s, Vh = np.linalg.svd(D1a)

    # now, get eigenvectors for D1a
    Ut, eigenvalues, Vt = sparseSVD(D1a)
    Ut = np.matrix(Ut)

    # then, an equivalent 2nd O term would be
    D2a = D1a ** 2

    # show we can recover D1a
    D1a_est = Ut.T * np.diag(eigenvalues) * Ut

    # Now, because D2a (& the target D1a) are symmetric:
    D1a_est = Ut.T * np.diag(eigenvalues ** 0.5) * Ut

    D = 2 * I - (np.roll(I, -1) + np.roll(I, 1))
    a = sparseSVD(-D)
    eigenvalues = np.matrix(a[1])
    Ut = np.matrix(a[0])
    Vt = np.matrix(a[2])
    orig = Ut.T * np.diag(np.array(eigenvalues).flatten()) * Vt

    Feigenvalues = np.diag(np.array(np.c_[eigenvalues, 0]).flatten())
    FUt = np.c_[Ut.T, np.zeros(Ut.shape[1])]
    # confirm: FUt * Feigenvalues * FUt.T ~= D

    # m is a 1st O difference matrix
    # with careful edge conditions
    # such that m.T * m = D2
    # D2 being a 2nd O difference matrix
    m = np.matrix(np.identity(100) - np.roll(np.identity(100), 1))
    m[-1, -1] = 0
    m[0, 0] = 1
    a = sparseSVD(m)
    eigenvalues = np.matrix(a[1])
    Ut = np.matrix(a[0])
    Vt = np.matrix(a[2])
    orig = Ut.T * np.diag(np.array(eigenvalues).flatten()) * Vt
    # Vt* Vt.T = I
    # Ut.T * Ut = I
    # ((Vt.T * (np.diag(np.array(eigenvalues).flatten())**2)) * Vt)
    # we see you get the same as m.T * m by squaring the eigenvalues


# from StackOverflow
# https://stackoverflow.com/questions/17115030/want-to-smooth-a-contour-from-a-masked-np.array

def smooth(u, mask):
    # set all 'masked' points to 0. so they aren't used in the smoothing
    r = u.filled(0.)
    m = mask
    a = 4*r[1:-1, 1:-1] + r[2:, 1:-1] + \
        r[:-2, 1:-1] + r[1:-1, 2:] + r[1:-1, :-2]
    b = 4*m[1:-1, 1:-1] + m[2:, 1:-1] + m[:-2, 1:-1] + m[1:-1, 2:] + \
        m[1:-1, :-2]  # a divisor that accounts for masked points
    b[b ==0] = 1.  # for avoiding divide by 0 error (region is masked so value doesn't matter)
    u[1:-1, 1:-1] = a/b


def smooth_masked_array(u):
    """ Use smooth() on the masked np.array """

    if not isinstance(u, np.ma.core.MaskedArray):
        raise ValueError("Expected masked np.array")

    m = u.mask

    # run the data through the smoothing filter a few times
    for i in range(10):
        smooth(u, m)

    return np.ma.array(u, mask=m)  # put together the mask and the data
