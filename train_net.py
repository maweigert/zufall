"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from keras.optimizers import Adam

from resunet_model import resunet_model
from create_samples import create_samples


def build_model(input_shape):
    return resunet_model(input_shape,
                         "relu",
                         1, 32, 3, 3, n_conv_per_depth=1)


if __name__ == '__main__':

    fname = "models/resunet.hdf5"
    n_epochs = 20

    print("creating samples...")
    X, Y = create_samples(200)

    print("building model...")
    model = build_model(X.shape[1:])


    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

    print("training for %s epochs..."%n_epochs)

    model.fit(X, Y, batch_size=64, epochs=n_epochs, validation_split=.1)

    model.save_weights(fname)
