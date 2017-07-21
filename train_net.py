"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from keras.optimizers import Adam

from resunet_model import resunet_model
from create_samples import create_stripes, create_letters


def build_model(input_shape):
    return resunet_model(input_shape,
                         "relu",
                         2, 32, 5, 5, n_conv_per_depth=2)


if __name__ == '__main__':

    fname = "models/resunet.hdf5"
    n_epochs = 200

    print("creating samples...")
    X, Y = create_samples(2000)

    print("building model...")
    model = build_model(X.shape[1:])


    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.0005))

    print("training for %s epochs..."%n_epochs)

    model.fit(X, Y, batch_size=64, epochs=n_epochs, validation_split=.1)

    model.save_weights(fname)
