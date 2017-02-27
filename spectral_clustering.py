import numpy as np
from sklearn.cluster import KMeans


def create_local_sim(A, k_hat):
    # Creating a dense similarity matrix with local scale
    # (sigma's) as shown in the slides, page 6
    # A: n*d data matrix
    # k_hat: as defined in the slide
    # Return value: n*n similarity matrix
    n = A.shape[0]
    assert k_hat < n
    a = (A*A).sum(axis = 1)
    a = np.tile(a, (n,1)).transpose()
    
    x = A.dot(A.transpose())
    
    distance = a - 2*x + a.transpose()
    distance = distance.astype(np.float64)
    
    distance_copy = np.zeros((n,n))
    distance_copy[:,:] = np.sqrt(distance[:,:])
    distance_copy.sort()
    kneighbor = distance_copy[:, k_hat]
    kneighbor = np.tile(kneighbor, (n,1)).transpose()
    tkneighbor = kneighbor.transpose()
    sigmas = kneighbor * tkneighbor
    W = pow(np.e, -distance/(2*sigmas))
    
    return W

def run(W, k, ncut):
    # Perform spectral clustering
    # W: n*n similarity matrix
    # k: number of clusters
    # ncut: perform ratio cut when ncut==False; normalized cut when ncut==True
    # Return value: n-vector that contains cluster assignments
    assert W.shape[0] == W.shape[1]
    n = W.shape[0]
    d = W.sum(axis = 1)
    D = np.diag(d)
    L = D - W
    if ncut == True:
        L = L / L.max()
    X, s, V = np.linalg.svd(L, full_matrices=False)
    evecs = X[:,(n-k):n]
    e_normed = evecs / np.tile(evecs.max(axis=1), (k,1)).transpose()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(e_normed)
    labels = kmeans.labels_
    return labels
