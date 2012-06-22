import numpy as np

def poly_to_gauss(A,B,C):
    sigma = np.sqrt(-1 / (2.0 * A))
    mu = B * sigma**2
    height = np.exp(C + 0.5 * mu**2 / sigma**2)
    return sigma, mu, height

def weighted_invert(x, y, weights=None, threshold=0):
    mask = y > threshold
    x,y = x[mask], y[mask]
    if weights is None:
        weights = y
    else:
        weights = weights[mask]

    d = np.log(y)
    G = np.ones((x.size, 3), dtype=np.float)
    G[:,0] = x**2
    G[:,1] = x

    model,_,_,_ = np.linalg.lstsq((G.T*weights**2).T, d*weights**2)
    return poly_to_gauss(*model)

