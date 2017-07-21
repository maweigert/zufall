"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import keras.backend as K


def create_samples(n_samples):
    n_size = 128
    Y = np.zeros((n_samples, 1, n_size, n_size))
    ss = (slice(n_size // 4, 3 * n_size // 4),) * 2
    Y[:, 0][ss] = 1.
    X = Y + .1 * np.random.uniform(-1, 1, Y.shape)

    if K.image_dim_ordering() == "tf":
        X = np.swapaxes(X, 1, -1)
        Y = np.swapaxes(Y, 1, -1)

    return X, Y


def create_samples(n_samples, n_size = 128, sigma_noise = 1.):


    # create some random stripe patterns
    x0 = np.arange(n_size)
    _Y0,_X0 = np.meshgrid(x0,x0, indexing = "ij")
    angles = np.random.normal(0, 1, (n_samples,2,1,1))
    angles *= 1./np.linalg.norm(angles, axis = 1, keepdims = True)

    Y = .5+.5*np.sin(angles[:,0]*_X0+angles[:,1]*_Y0)[:, np.newaxis,:,:]
    X = Y + sigma_noise * np.random.normal(-1, 1, Y.shape)

    if K.image_dim_ordering() == "tf":
        X = np.swapaxes(X, 1, -1)
        Y = np.swapaxes(Y, 1, -1)

    return X, Y


if __name__ == '__main__':

    X,Y = create_samples(10)