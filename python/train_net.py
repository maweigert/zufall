"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from keras.optimizers import Adam
from deeptools.losses import mse_weighted
from resunet_model import resunet_model
from create_samples import create_sample_words


def build_model(input_shape):
    return resunet_model(input_shape,
                         "relu",
                         2, 32, 5, 5, n_conv_per_depth=2)


if __name__ == '__main__':
    n_epochs = 50
    sigma_noise = 1.2
    fname = "models/resunet_%.2f.hdf5"%sigma_noise

    print("creating samples...")
    X, Y = create_sample_words(2000, sigma_noise=sigma_noise)

    thresh = .5
    freq = np.mean(Y>thresh)

    print("building model...")
    model = build_model((None,None,1))

    model.compile(
        #loss="mean_squared_error",
        loss = mse_weighted(thresh,1./(1.-freq), 1./freq),
        optimizer=Adam(lr=0.0001))

    print("training for %s epochs..." % n_epochs)

    model.fit(X, Y, batch_size=64, epochs=n_epochs, validation_split=.1)

    model.save(fname, overwrite=True)
