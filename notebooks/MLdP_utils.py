# MLdP_utils.py
# Utility functions from "Machine Learning for Asset Managers"

# Required Imports:
# MLdP_utils.py

# Core
import numpy as np
import pandas as pd

# Stats and math
from scipy.optimize import minimize
from scipy.linalg import block_diag
import scipy.stats as ss

# ML tools
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, mutual_info_score
from sklearn.neighbors import KernelDensity

# Optional 
from sklearn.utils import check_random_state


# Chapter 2:

#--------------------------------------------------- 
def mpPDF(var, q, pts):
    # Marcenko-Pastur PDF
    # q = T / N
    eMin = var * (1 - (1. / q) ** 0.5) ** 2
    eMax = var * (1 + (1. / q) ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index=eVal)
    return pdf

#---------------------------------------------------
def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

#---------------------------------------------------
def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    values = np.exp(logProb)
    index = x.ravel().astype(float)
    pdf = pd.Series(values, index)
    return pdf

def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)  # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols))  # full rank
    return cov

#---------------------------------------------------
def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr

#---------------------------------------------------
def errPDFs(var, eVal, q, bWidth, pts=1000):
    # Fit error
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse

#---------------------------------------------------
def findMaxEval(eVal, q, bWidth):
    # Use a lambda that extracts the float from the array
    out = minimize(lambda var: errPDFs(var[0], eVal, q, bWidth),
                   x0=[0.5], bounds=[(1E-5, 1 - 1E-5)])
    if out['success']:
        var = out['x'][0]
    else:
        var = 1.0
    eMax = var * (1 + (1. / q) ** 0.5) ** 2
    return eMax, var
#---------------------------------------------------

def denoisedCorr(eVal, eVec, nFacts):
    # Step 1: Copy eigenvalues
    eVal_ = np.diag(eVal).copy()

    # Step 2: Average the noise eigenvalues and replace them
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)

    # Step 3: Reconstruct the matrix
    corr1 = np.dot(eVec, np.diag(eVal_)).dot(eVec.T)

    # Step 4: Rescale to make diagonal = 1 (turn into correlation matrix)
    corr1 = cov2corr(corr1)

    return corr1
#---------------------------------------------------

def denoisedCorr2(eVal, eVec, nFacts, alpha=0):
    # Remove noise from corr through targeted shrinkage

    # Signal eigenvalues and eigenvectors
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]

    # Noise eigenvalues and eigenvectors
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]

    # Reconstruct signal and noise components
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)

    # Apply targeted shrinkage to noise
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    
    return corr2

# Convert correlation matrix to covariance matrix
def corr2cov(corr, std):
    return np.outer(std, std) * corr

# Create a block-diagonal correlation matrix
def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    np.fill_diagonal(block, 1)  # set diagonal to 1
    corr = block_diag(*([block] * nBlocks))
    return corr

#---------------------------------------------------
# Create the true covariance matrix and mean vector
def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)

    # Shuffle rows/columns to avoid block structure bias
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)

    # Generate random volatilities (std dev)
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])

    # Convert to covariance matrix
    cov0 = corr2cov(corr0, std0)

    # Generate random mean vector
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)

    return mu0, cov0

def simCovMu(mu0, cov0, nObs, shrink=False):
    # Simulate multivariate returns and estimate mean/covariance
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    
    # Sample mean vector
    mu1 = x.mean(axis=0).reshape(-1, 1)
    
    # Sample covariance matrix
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=False)
    
    return mu1, cov1

def deNoiseCov(cov0, q, bWidth):
    # Step 1: Convert covariance matrix to correlation matrix
    corr0 = cov2corr(cov0)

    # Step 2: Perform PCA
    eVal0, eVec0 = getPCA(corr0)

    # Step 3: Find Marcenko–Pastur threshold and number of factors
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)

    # Step 4: Denoise the correlation matrix
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)

    # Step 5: Rescale denoised correlation back to covariance
    cov1 = corr2cov(corr1, np.sqrt(np.diag(cov0)))

    return cov1

#---------------------------------------------------
def optPort(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

#---------------------------------------------------

# Chapter 3:

def varInfo(x, y, bins, norm=False):
    """
    Compute the variation of information (VI) between two variables x and y.
    
    Parameters:
        x (array-like): First input vector.
        y (array-like): Second input vector.
        bins (int): Number of bins to use for histograms.
        norm (bool): Whether to normalize VI by the joint entropy.
    
    Returns:
        float: Variation of information (normalized if norm=True).
    """
    # Joint histogram
    cXY = np.histogram2d(x, y, bins)[0]
    
    # Mutual information
    iXY = mutual_info_score(None, None, contingency=cXY)
    
    # Marginal entropies
    hX = ss.entropy(np.histogram(x, bins)[0])
    hY = ss.entropy(np.histogram(y, bins)[0])
    
    # Variation of information
    vXY = hX + hY - 2 * iXY
    
    if norm:
        hXY = hX + hY - iXY  # Joint entropy
        vXY /= hXY           # Normalized VI
    
    return vXY

#---------------------------------------------------

# Chapter 4:

def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    x = ((1 - corr0.fillna(0)) / 2.)**.5
    silh_best, kmeans_best = pd.Series(), None
    for _ in range(n_init):
        for i in range(2, maxNumClusters + 1):
            kmeans = KMeans(n_clusters=i, n_init=1, random_state=42)
            kmeans = kmeans.fit(x)
            silh = silhouette_samples(x, kmeans.labels_)
            score = np.mean(silh) / np.std(silh)
            if kmeans_best is None or np.isnan(score) or score > np.mean(silh_best) / np.std(silh_best):
                silh_best, kmeans_best = silh, kmeans
    newIdx = np.argsort(kmeans_best.labels_)
    corr1 = corr0.iloc[newIdx, :].iloc[:, newIdx]
    clstrs = {i: corr0.columns[np.where(kmeans_best.labels_ == i)[0]].tolist()
              for i in np.unique(kmeans_best.labels_)}
    silh_series = pd.Series(silh_best, index=x.index)
    return corr1, clstrs, silh_series

#---------------------------------------------------

def makeNewOutputs(corr0, clstrs, clstrs2):
    clstrsNew = {}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])

    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]

    x = ((1 - corr0.fillna(0)) / 2.) ** 0.5
    kmeans_labels = np.zeros(len(x.columns))
    for i in clstrsNew.keys():
        idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i

    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    return corrNew, clstrsNew, silhNew

#---------------------------------------------------
def clusterKMeansTop(corr0, maxNumClusters=None, n_init=10):
    if maxNumClusters is None:
        maxNumClusters = corr0.shape[1] - 1

    corr1, clstrs, silh = clusterKMeansBase(
        corr0,
        maxNumClusters=min(maxNumClusters, corr0.shape[1] - 1),
        n_init=n_init
    )

    clusterTstats = {
        i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]])
        for i in clstrs.keys()
    }
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)

    redoClusters = [
        i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean
    ]

    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])

        corr2, clstrs2, silh2 = clusterKMeansTop(
            corrTmp,
            maxNumClusters=min(maxNumClusters, corrTmp.shape[1] - 1),
            n_init=n_init
        )

        corrNew, clstrsNew, silhNew = makeNewOutputs(
            corr0,
            {i: clstrs[i] for i in clstrs.keys() if i not in redoClusters},
            clstrs2
        )

        newTstatMean = np.mean([
            np.mean(silhNew[clstrsNew[i]]) / np.std(silhNew[clstrsNew[i]])
            for i in clstrsNew.keys()
        ])

        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew

#---------------------------------------------------
def getCovSub(nObs, nCols, sigma, random_state=None):
    # Sub correlation matrix
    rng = check_random_state(random_state)
    if nCols == 1:
        return np.ones((1, 1))
    ar0 = rng.normal(size=(nObs, 1))
    ar0 = np.repeat(ar0, nCols, axis=1)
    ar0 += rng.normal(scale=sigma, size=ar0.shape)
    ar0 = np.cov(ar0, rowvar=False)
    return ar0

#---------------------------------------------------
def getRndBlockCov(nCols, nBlocks, minBlockSize=1, sigma=1., random_state=None):
    # Generate a block random correlation matrix
    rng = check_random_state(random_state)
    parts = rng.choice(
        range(1, nCols - (minBlockSize - 1) * nBlocks),
        nBlocks - 1,
        replace=False
    )
    parts.sort()
    parts = np.append(parts, nCols - (minBlockSize - 1) * nBlocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + minBlockSize

    cov = None
    for nCols_ in parts:
        cov_ = getCovSub(
            int(max(nCols_ * (nCols_ + 1) / 2., 100)),
            nCols_,
            sigma,
            random_state=rng
        )
        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)
    return cov

#---------------------------------------------------
def randomBlockCorr(nCols, nBlocks, random_state=None, minBlockSize=1):
    # Form block corr
    rng = check_random_state(random_state)
    cov0 = getRndBlockCov(
        nCols,
        nBlocks,
        minBlockSize=minBlockSize,
        sigma=0.5,
        random_state=rng
    )
    cov1 = getRndBlockCov(
        nCols,
        1,
        minBlockSize=minBlockSize,
        sigma=1.0,
        random_state=rng
    )  # add noise
    cov0+=cov1 
    corr0=cov2corr(cov0) 
    corr0=pd.DataFrame(corr0) 
    return corr0

#---------------------------------------------------

# Chapter 5: