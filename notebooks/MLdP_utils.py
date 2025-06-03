# MLdP_utils.py
# Utility functions from "Machine Learning for Asset Managers"

# Required Imports:

# Core
import numpy as np
import pandas as pd

# Stats and math
from scipy.optimize import minimize
from scipy.linalg import block_diag
import scipy.stats as ss
import statsmodels.api as sm

# ML tools
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, mutual_info_score, log_loss
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

# Optional 
from sklearn.utils import check_random_state


# Chapter 2:

#--------------------------------------------------- 
def mpPDF(var, q, pts):
    """
    Computes the Marčenko–Pastur Probability Density Function (PDF).

    The Marčenko–Pastur distribution models the theoretical distribution of eigenvalues 
    for large random covariance matrices. It is used in Random Matrix Theory to identify 
    the spectrum of noise in empirical correlation or covariance matrices.

    Parameters:
    ----------
    var : float
        The variance of the distribution (typically 1 for standardized returns).
    q : float
        The ratio T/N, where T is the number of observations and N is the number of variables (q > 1).
    pts : int
        The number of points to discretize the PDF for plotting or numerical comparison.

    Returns:
    -------
    pdf : pandas.Series
        The Marčenko–Pastur PDF evaluated at 'pts' evenly spaced eigenvalues between the theoretical minimum and maximum.
        The index contains the eigenvalues, and the values contain the corresponding PDF values.
    """
    eMin = var * (1 - (1. / q) ** 0.5) ** 2
    eMax = var * (1 + (1. / q) ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index=eVal)
    return pdf

#---------------------------------------------------
def getPCA(matrix):
    """
    Performs Principal Component Analysis (PCA) on a Hermitian matrix.

    This function computes the eigenvalues and eigenvectors of a symmetric (Hermitian) matrix,
    which is typically a covariance or correlation matrix in finance and machine learning.
    The eigenvalues are sorted in descending order, and the eigenvectors are reordered accordingly.

    Parameters:
    ----------
    matrix : numpy.ndarray
        A symmetric (Hermitian) matrix, such as a covariance or correlation matrix.

    Returns:
    -------
    eVal : numpy.ndarray
        A diagonal matrix containing the sorted eigenvalues in descending order.
    eVec : numpy.ndarray
        The matrix of eigenvectors corresponding to the sorted eigenvalues.
    """
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

#---------------------------------------------------
def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    """
    Fits a Kernel Density Estimator (KDE) to a set of observations.

    This function estimates the probability density function (PDF) of a random variable 
    using kernel density estimation. It allows evaluation of the fitted density at a 
    specified set of points `x`.

    Parameters:
    ----------
    obs : numpy.ndarray
        A 1D array of observed values to which the KDE will be fitted.
    bWidth : float, optional
        Bandwidth (smoothing parameter) of the kernel. Default is 0.25.
    kernel : str, optional
        The kernel to use for the KDE. Default is 'gaussian'. Other options include 
        'tophat', 'epanechnikov', etc., as supported by sklearn's KernelDensity.
    x : numpy.ndarray, optional
        Points at which to evaluate the fitted PDF. If None, uses the unique values of `obs`.

    Returns:
    -------
    pdf : pandas.Series
        The estimated probability density function evaluated at the values in `x`.
        The index is the `x` values and the values are the estimated densities.
    """
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

#-------------------------------------------------------

def getRndCov(nCols, nFacts):
    """
    Generates a random full-rank covariance matrix using a factor model.

    The function simulates a random covariance matrix by creating a factor-loading matrix 
    and adding a diagonal matrix of idiosyncratic noise to ensure full rank.

    Parameters:
    ----------
    nCols : int
        Number of variables (i.e., the dimension of the covariance matrix).
    nFacts : int
        Number of underlying latent factors to simulate (must be <= nCols).

    Returns:
    -------
    cov : numpy.ndarray
        A symmetric, positive-definite covariance matrix of shape (nCols, nCols).
    """
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)  # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols))  # full rank
    return cov

#---------------------------------------------------
def cov2corr(cov):
    """
    Convert a covariance matrix to a correlation matrix.
    """
    
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr

#---------------------------------------------------
def errPDFs(var, eVal, q, bWidth, pts=1000):
    """
    Computes the sum of squared errors (SSE) between the theoretical Marčenko–Pastur PDF 
    and the empirical PDF estimated via Kernel Density Estimation (KDE).

    This function is useful for assessing how well a set of eigenvalues fits the 
    theoretical distribution expected under Random Matrix Theory.

    Parameters:
    ----------
    var : float
        Variance of the underlying distribution (e.g., asset returns).
    eVal : numpy.ndarray
        Array of empirical eigenvalues from a covariance/correlation matrix.
    q : float
        The ratio T/N, where T is the number of observations and N is the number of variables (q > 1).
    bWidth : float
        Bandwidth for the KDE used to estimate the empirical PDF.
    pts : int, optional
        Number of evaluation points for the theoretical Marčenko–Pastur PDF (default is 1000).

    Returns:
    -------
    sse : float
        The sum of squared errors between the theoretical and empirical PDFs.
    """
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse

#---------------------------------------------------
def findMaxEval(eVal, q, bWidth):
    """
    Estimates the maximum theoretical eigenvalue (eMax) using Marčenko–Pastur theory 
    by minimizing the error between the empirical and theoretical PDFs.

    This function finds the variance parameter that minimizes the sum of squared errors 
    between the KDE-fitted empirical eigenvalue distribution and the Marčenko–Pastur distribution.
    It then computes the theoretical maximum eigenvalue based on that optimal variance.

    Parameters:
    ----------
    eVal : numpy.ndarray
        Array of empirical eigenvalues from a covariance/correlation matrix.
    q : float
        The ratio T/N, where T is the number of observations and N is the number of variables (q > 1).
    bWidth : float
        Bandwidth used for Kernel Density Estimation of the empirical PDF.

    Returns:
    -------
    eMax : float
        The estimated maximum eigenvalue according to the Marčenko–Pastur distribution.
    var : float
        The variance that minimizes the error between the theoretical and empirical PDFs.
    """

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
    """
    Constructs a denoised correlation matrix using eigenvalue clipping.

    This method retains the top `nFacts` principal components (interpreted as signal) 
    and replaces the remaining (noise) eigenvalues with their average. The matrix is then 
    reconstructed and rescaled to a proper correlation matrix.

    This requires you first find as follows:
        eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=0.01)
        nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)

    Parameters:
    ----------
    eVal : numpy.ndarray
        A diagonal matrix (2D array) of eigenvalues from PCA on a correlation matrix.
    eVec : numpy.ndarray
        A matrix of eigenvectors corresponding to `eVal`.
    nFacts : int
        The number of dominant factors (signal) to retain. Remaining components are treated as noise.

    Returns:
    -------
    corr1 : numpy.ndarray
        The denoised correlation matrix, rescaled to have ones on the diagonal.
    """
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
    """
    Constructs a denoised correlation matrix using targeted shrinkage on noise components.

    This method separates the signal and noise components of a correlation matrix using PCA. 
    The signal is reconstructed from the top `nFacts` eigenvalues, while the noise is reconstructed 
    from the remaining components. A shrinkage parameter `alpha` controls how much of the noise 
    structure is retained versus replaced by its diagonal (idiosyncratic risk).

    Parameters:
    ----------
    eVal : numpy.ndarray
        A diagonal matrix (2D array) of eigenvalues from PCA on a correlation matrix.
    eVec : numpy.ndarray
        A matrix of eigenvectors corresponding to `eVal`.
    nFacts : int
        Number of principal components (signal factors) to retain.
    alpha : float, optional (default=0)
        Shrinkage parameter in [0, 1] applied to the noise:
        - 0: keep only the diagonal of the noise (pure idiosyncratic risk),
        - 1: fully reconstruct the noise matrix structure.

    Returns:
    -------
    corr2 : numpy.ndarray
        The denoised correlation matrix combining the signal and shrunken noise components.
    """

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
#---------------------------------------------------

# Convert correlation matrix to covariance matrix
def corr2cov(corr, std):
    return np.outer(std, std) * corr

#---------------------------------------------------
# Create a block-diagonal correlation matrix
def formBlockMatrix(nBlocks, bSize, bCorr):
    """
    Creates a block-diagonal correlation matrix with uniform intra-block correlation.

    Each block represents a cluster of variables (e.g., assets) that are 
    equally correlated with one another. Blocks are uncorrelated with each other.

    Parameters:
    ----------
    nBlocks : int
        Number of blocks (clusters) in the matrix.
    bSize : int
        Size of each block (number of variables per cluster).
    bCorr : float
        Intra-block correlation value (must be in [0, 1)).

    Returns:
    -------
    corr : numpy.ndarray
        A block-diagonal correlation matrix of shape (nBlocks * bSize, nBlocks * bSize),
        with 1s on the diagonal and `bCorr` off-diagonal within each block.
    """
    
    block = np.ones((bSize, bSize)) * bCorr
    np.fill_diagonal(block, 1)  # set diagonal to 1
    corr = block_diag(*([block] * nBlocks))
    return corr

#---------------------------------------------------
# Create the true covariance matrix and mean vector
def formTrueMatrix(nBlocks, bSize, bCorr):
    """
    Constructs a realistic synthetic covariance matrix and mean vector from a block-diagonal correlation structure.

    This function generates a block-structured correlation matrix with random permutations, 
    random volatilities, and a random mean vector—useful for simulating asset returns in 
    portfolio optimization or clustering experiments.

    Parameters:
    ----------
    nBlocks : int
        Number of blocks (clusters) in the correlation matrix.
    bSize : int
        Number of variables (e.g., assets) per block.
    bCorr : float
        Intra-block correlation value (between 0 and 1).

    Returns:
    -------
    mu0 : numpy.ndarray
        A column vector of simulated expected returns (shape: [nAssets, 1]).
    cov0 : numpy.ndarray
        A full-rank covariance matrix (shape: [nAssets, nAssets]) derived from the 
        block-structured correlation matrix and random volatilities.
    """
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

# -----------------------------------------------------
def simCovMu(mu0, cov0, nObs, shrink=False):
    """
    Simulates multivariate returns and estimates their sample mean and covariance.

    This function generates synthetic data from a multivariate normal distribution 
    using a true mean vector and covariance matrix. It then computes the sample 
    estimates of the mean and covariance, optionally applying Ledoit-Wolf shrinkage.

    Used in pair with 'formTrueMatrix' to create a realistic dataset for testing.

    Parameters:
    ----------
    mu0 : numpy.ndarray
        True mean vector of shape [nAssets, 1].
    cov0 : numpy.ndarray
        True covariance matrix of shape [nAssets, nAssets].
    nObs : int
        Number of simulated observations (e.g., days of returns).
    shrink : bool, optional
        Whether to apply Ledoit-Wolf shrinkage to the sample covariance matrix (default is False).

    Returns:
    -------
    mu1 : numpy.ndarray
        Estimated sample mean vector (shape: [nAssets, 1]).
    cov1 : numpy.ndarray
        Estimated sample covariance matrix (either empirical or shrinkage-adjusted).
    """
    
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    
    # Sample mean vector
    mu1 = x.mean(axis=0).reshape(-1, 1)
    
    # Sample covariance matrix
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=False)
    
    return mu1, cov1

#---------------------------------------------------
def deNoiseCov(cov0, q, bWidth):
    """
    Applies Random Matrix Theory (RMT) to denoise a covariance matrix via 
    Marčenko–Pastur eigenvalue filtering and PCA-based reconstruction.

    This process improves the stability and estimation accuracy of the covariance 
    matrix by removing noisy components and retaining only significant eigenmodes.

    Parameters:
    ----------
    cov0 : numpy.ndarray
        Empirical covariance matrix (shape: [nAssets, nAssets]).
    q : float
        The ratio T/N, where T is the number of observations and N is the number of variables (q > 1).
    bWidth : float
        Bandwidth parameter for kernel density estimation used in Marčenko–Pastur fitting.

    Returns:
    -------
    cov1 : numpy.ndarray
        Denoised covariance matrix reconstructed from the filtered correlation matrix 
        and the original standard deviations.
    """
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
    """
    Computes the minimum variance portfolio weights, optionally with a return target.

    If `mu` is provided, the function computes the mean-variance optimal (Markowitz) 
    portfolio that maximizes the Sharpe ratio (with unit sum constraint). If `mu` is 
    not provided, it defaults to the global minimum variance (GMV) portfolio.

    Parameters:
    ----------
    cov : numpy.ndarray
        Covariance matrix of asset returns (shape: [nAssets, nAssets]).
    mu : numpy.ndarray, optional
        Expected return vector (shape: [nAssets, 1]). If None, assumes all assets 
        have equal expected return (i.e., computes the GMV portfolio).

    Returns:
    -------
    w : numpy.ndarray
        Optimal portfolio weights (shape: [nAssets, 1]) that sum to 1.
    """
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
    """
    Applies K-Means clustering to a correlation matrix to identify clusters of similar assets.

    The correlation matrix is transformed into a distance matrix using the angular distance formula,
    and K-Means clustering is applied multiple times to find the most stable and well-separated clusters
    using the silhouette score as a criterion.

    Parameters:
    ----------
    corr0 : pandas.DataFrame
        The input correlation matrix (symmetric, with values in [-1, 1]).
    maxNumClusters : int, optional
        The maximum number of clusters to test (default is 10).
    n_init : int, optional
        Number of random initializations to run the clustering process for robustness (default is 10).

    Returns:
    -------
    corr1 : pandas.DataFrame
        The reordered correlation matrix according to the best clustering assignment.
    clstrs : dict
        A dictionary mapping each cluster label to a list of asset names (column labels in `corr0`).
    silh_series : pandas.Series
        A series of silhouette scores for each observation, indicating cluster assignment quality.
    """

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
    """
    Merges two cluster dictionaries and reorders a correlation matrix accordingly, 
    computing the updated silhouette scores for the new clustering structure.

    This is useful when combining clustering results from two different runs 
    (e.g., different parameter sets) and evaluating the quality of the merged clustering.

    Parameters:
    ----------
    corr0 : pandas.DataFrame
        Original correlation matrix with asset labels as both index and columns.
    clstrs : dict
        First dictionary of clusters; keys are cluster IDs, values are lists of asset names.
    clstrs2 : dict
        Second dictionary of clusters to merge with `clstrs`.

    Returns:
    -------
    corrNew : pandas.DataFrame
        Correlation matrix reordered based on the merged cluster assignments.
    clstrsNew : dict
        Combined dictionary of clusters, with reindexed cluster labels.
    silhNew : pandas.Series
        Silhouette scores for each asset in the new clustering structure.
    """
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
    """
    Recursively applies and refines K-Means clustering on a correlation matrix to improve cluster quality.

    This function starts with a base K-Means clustering and evaluates the clustering quality 
    using a t-statistic (mean / std of silhouette scores) for each cluster. Clusters with below-average 
    t-statistics are recursively reclustered, and the process continues until no significant improvement 
    is found.

    Parameters:
    ----------
    corr0 : pandas.DataFrame
        Input correlation matrix with asset labels as both index and columns.
    maxNumClusters : int, optional
        Maximum number of clusters to try. If None, defaults to the number of assets minus one.
    n_init : int, optional
        Number of random initializations for K-Means to improve robustness. Default is 10.

    Returns:
    -------
    corrFinal : pandas.DataFrame
        Final reordered correlation matrix based on refined clustering.
    clstrsFinal : dict
        Final dictionary of clusters after recursive improvement.
    silhFinal : pandas.Series
        Final silhouette scores for all assets under the refined clustering.
    """
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
    """
    Generates a synthetic covariance matrix with a shared underlying signal and controlled noise.

    This function simulates `nCols` time series that share a common latent factor 
    (perfectly correlated signal), plus additive Gaussian noise with standard deviation `sigma`.
    It is useful for testing denoising or clustering techniques.

    Parameters:
    ----------
    nObs : int
        Number of observations (e.g., time points).
    nCols : int
        Number of variables (e.g., assets or features).
    sigma : float
        Standard deviation of the added noise (controls correlation strength).
    random_state : int or np.random.RandomState, optional
        Seed or random number generator for reproducibility.

    Returns:
    -------
    ar0 : numpy.ndarray
        A covariance matrix of shape (nCols, nCols) derived from the noisy, shared-signal time series.
    """
    
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
    """
    Generates a block-structured random covariance matrix with noisy sub-blocks.

    Each block simulates a group of correlated variables (e.g., assets) sharing an underlying 
    signal with added Gaussian noise. Block sizes are randomly determined but constrained to 
    be at least `minBlockSize`.

    Parameters:
    ----------
    nCols : int
        Total number of variables (e.g., assets).
    nBlocks : int
        Number of correlation blocks to generate.
    minBlockSize : int, optional
        Minimum number of variables per block (default is 1).
    sigma : float, optional
        Standard deviation of noise in each sub-block (higher = lower intra-block correlation).
    random_state : int or np.random.RandomState, optional
        Seed or generator for reproducibility.

    Returns:
    -------
    cov : numpy.ndarray
        A block-diagonal covariance matrix of shape (nCols, nCols), composed of randomly 
        sized sub-blocks with correlated noise structures.
    """
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
    """
    Generates a randomized block-structured correlation matrix with added noise.

    This function creates a synthetic correlation matrix composed of correlated blocks (clusters)
    using `getRndBlockCov`, and then injects additional random noise to make the structure more realistic.
    The resulting matrix is normalized into a proper correlation matrix.

    Parameters:
    ----------
    nCols : int
        Total number of variables (e.g., assets or features).
    nBlocks : int
        Number of underlying correlation blocks to generate.
    random_state : int or np.random.RandomState, optional
        Seed or random number generator for reproducibility.
    minBlockSize : int, optional
        Minimum number of variables per block (default is 1).

    Returns:
    -------
    corr0 : pandas.DataFrame
        A symmetric, positive semi-definite correlation matrix with block structure and noise,
        formatted as a pandas DataFrame.
    """
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

#---------------------------------------------------
def tValLinR(close):
    """
    Computes the t-value of the slope coefficient from a linear regression on a time series.

    This function fits an Ordinary Least Squares (OLS) regression of the input `close` values 
    against time (trend) and returns the t-statistic of the slope. A high absolute t-value 
    indicates a statistically significant trend.

    Parameters:
    ----------
    close : array-like or pandas.Series
        Time series of values (e.g., asset closing prices).

    Returns:
    -------
    t_val : float
        The t-statistic for the slope coefficient of the linear regression.
    """
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm.OLS(close, x).fit()
    return ols.tvalues[1]

#---------------------------------------------------

def getBinsFromTrend(molecule, close, span):
    '''
    Derive labels from the sign of t-value of linear trend.
    Output includes:
    - t1: End time for the identified trend
    - tVal: t-value associated with the estimated trend coefficient
    - bin: Sign of the trend
    '''
    out = pd.DataFrame(index=molecule, columns=['t1', 'tVal', 'bin'])
    hrzns = range(*span)

    for dt0 in molecule:
        df0 = pd.Series()
        iloc0 = close.index.get_loc(dt0)
        if iloc0 + max(hrzns) > close.shape[0]:
            continue

        for hrzn in hrzns:
            dt1 = close.index[iloc0 + hrzn - 1]
            df1 = close.loc[dt0:dt1]
            df0.loc[dt1] = tValLinR(df1.values)

        dt1 = df0.replace([np.inf, -np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, ['t1', 'tVal', 'bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])

    # Prevent leakage and cast types
    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    
    return out.dropna(subset=['bin'])

#---------------------------------------------------

# Chapter 6:

#---------------------------------------------------

def getTestData(n_features=100, n_informative=25, n_redundant=25,
                n_samples=10000, random_state=0, sigmaStd=0.0):
    """
    Generate a synthetic classification dataset with informative,
    non-informative, and redundant features.
    """
    np.random.seed(random_state)

    # Generate informative + noise (non-redundant) features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state
    )

    # Name informative and noise features
    cols = ['I_' + str(i) for i in range(n_informative)]
    cols += ['N_' + str(i) for i in range(n_features - n_informative - n_redundant)]

    # Convert to DataFrame/Series
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)

    # Add redundant features as noisy copies of informative ones
    selected = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(selected):
        X['R_' + str(k)] = X['I_' + str(j)] + np.random.normal(scale=sigmaStd, size=X.shape[0])

    return X, y

#---------------------------------------------------

def featImpMDI(fit, featNames):
    """
    Compute feature importance using the Mean Decrease in Impurity (MDI) method
    based on a Bagging ensemble of decision trees.

    Parameters:
    - fit: fitted BaggingClassifier model
    - featNames: list of feature names

    Returns:
    - DataFrame with mean and standard error of feature importances
    """
    # Extract feature importances from each tree
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames

    # Replace zeros with NaN to avoid skewing the average
    df0 = df0.replace(0, np.nan)

    # Compute mean and standard error (CLT)
    imp = pd.concat({
        'mean': df0.mean(),
        'std': df0.std() * df0.shape[0]**-0.5
    }, axis=1)

    # Normalize by sum of means
    imp /= imp['mean'].sum()

    return imp

#---------------------------------------------------

def featImpMDA(clf, X, y, n_splits=10):
    """
    Feature importance based on Mean Decrease in Accuracy (MDA),
    using out-of-sample log-loss from K-fold cross-validation.
    
    Parameters:
    - clf: classifier with predict_proba method
    - X: DataFrame of features
    - y: Series of labels
    - n_splits: number of cross-validation splits

    Returns:
    - DataFrame with mean and std feature importance scores
    """
    cvGen = KFold(n_splits=n_splits)
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0 = X.iloc[train], y.iloc[train]
        X1, y1 = X.iloc[test], y.iloc[test]

        fit = clf.fit(X0, y0)
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # shuffle one feature
            prob_shuffled = fit.predict_proba(X1_)
            scr1.loc[i, j] = -log_loss(y1, prob_shuffled, labels=clf.classes_)

    # Compute relative decrease in performance
    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)

    # Aggregate with CLT
    imp = pd.concat({
        'mean': imp.mean(),
        'std': imp.std() * imp.shape[0]**-0.5
    }, axis=1)

    return imp

#---------------------------------------------------

def groupMeanStd(df0, clstrs):
    """
    Compute mean and standard error of grouped columns in df0.

    Parameters:
    - df0: DataFrame of feature importances (rows = trees, columns = features)
    - clstrs: Series or dict mapping group index to feature list

    Returns:
    - DataFrame with 'mean' and 'std' of group-wise importance
    """
    out = pd.DataFrame(columns=['mean', 'std'])

    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)  # sum importance of features in group
        out.loc['C_' + str(i), 'mean'] = df1.mean()
        out.loc['C_' + str(i), 'std'] = df1.std() * df1.shape[0]**-0.5  # standard error

    return out

# --------------------------------------------
def featImpMDI_Clustered(fit, featNames, clstrs):
    """
    Compute MDI feature importance at the cluster/group level.

    Parameters:
    - fit: Fitted BaggingClassifier or RandomForest model
    - featNames: List of feature names
    - clstrs: Dictionary mapping group index to list of feature names in that group

    Returns:
    - DataFrame with normalized mean and standard error of importances per cluster
    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # replace zeros to avoid bias from inactive splits

    imp = groupMeanStd(df0, clstrs)
    imp /= imp['mean'].sum()  # normalize total importance to 1

    return imp

# --------------------------------------------

def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10):
    """
    Compute cluster-level MDA feature importance via cross-validated log-loss.

    Parameters:
    - clf: Classifier with predict_proba method
    - X: DataFrame of features
    - y: Target labels
    - clstrs: Dictionary mapping cluster names to lists of feature names
    - n_splits: Number of K-fold splits

    Returns:
    - DataFrame of normalized mean and standard error for each cluster
    """
    cvGen = KFold(n_splits=n_splits)
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=clstrs.keys())

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0 = X.iloc[train], y.iloc[train]
        X1, y1 = X.iloc[test], y.iloc[test]

        fit = clf.fit(X0, y0)
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)

        for j in scr1.columns:  # each cluster
            X1_ = X1.copy(deep=True)
            for k in clstrs[j]:  # each feature in the cluster
                np.random.shuffle(X1_[k].values)  # shuffle feature
            prob_shuffled = fit.predict_proba(X1_)
            scr1.loc[i, j] = -log_loss(y1, prob_shuffled, labels=clf.classes_)

    # Compute relative importance and normalize
    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)

    imp = pd.concat({
        'mean': imp.mean(),
        'std': imp.std() * imp.shape[0]**-0.5
    }, axis=1)

    # Prefix cluster names with 'C_'
    imp.index = ['C_' + str(i) for i in imp.index]

    return imp

#---------------------------------------------------

# Chapter 7:

#---------------------------------------------------

def minVarPort(cov):
    """
    Compute minimum variance portfolio weights.

    Parameters
    ----------
    cov : pd.DataFrame or np.ndarray
        Covariance matrix of asset returns.

    Returns
    -------
    np.ndarray
        Optimal weights that minimize portfolio variance.
    """
    cov = np.array(cov)
    inv_cov = np.linalg.pinv(cov)  # Use pseudo-inverse for numerical stability
    ones = np.ones(cov.shape[0])
    w = inv_cov @ ones
    w /= ones.T @ inv_cov @ ones
    return w.reshape(-1, 1)

#---------------------------------------------------
def optPort_nco(cov, mu=None, maxNumClusters=None):
    """
    Computes the Nested Clustered Optimization (NCO) portfolio weights. 

    Needs to be denoised first

    Parameters:
    - cov: Covariance matrix (DataFrame)
    - mu: Expected returns (optional, as np.ndarray or pd.Series)
    - maxNumClusters: Maximum number of clusters for k-means clustering

    Returns:
    - nco: Optimal portfolio weights (as a column vector)
    """
    cov = pd.DataFrame(cov)

    if mu is not None:
        mu = pd.Series(mu[:, 0])

    # Step 1: Correlation clustering
    corr1 = cov2corr(cov)
    corr1, clstrs, _ = clusterKMeansBase(corr1, maxNumClusters=maxNumClusters, n_init=10)

    # Step 2: Intra-cluster optimization
    wIntra = pd.DataFrame(0.0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_ = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None:
            mu_ = None
        else:
            mu_ = mu.loc[clstrs[i]].values.reshape(-1, 1)

        wIntra.loc[clstrs[i], i] = optPort(cov_, mu_).flatten()

    # Step 3: Inter-cluster optimization
    cov_ = wIntra.T @ cov.values @ wIntra
    mu_ = None if mu is None else wIntra.T @ mu.values.reshape(-1, 1)
    wInter = pd.Series(optPort(cov_, mu_).flatten(), index=cov_.index)

    # Step 4: Combine weights
    nco = wIntra.mul(wInter, axis=1).sum(axis=1).values.reshape(-1, 1)
    
    return nco
#---------------------------------------------------