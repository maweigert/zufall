"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from keras.optimizers import Adam

from resunet_model import resunet_model
from create_samples import create_sample_words


def build_model(input_shape):
    return resunet_model(input_shape,
                         "relu",
                         2, 32, 9, 9, n_conv_per_depth=2)


if __name__ == '__main__':
    fname = "models/resunet.hdf5"
    n_epochs = 60

    print("creating samples...")
    X, Y = create_sample_words(2000, sigma_noise=1.9)


    print("building model...")
    model = build_model((None,None,1))

    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.0001))

    print("training for %s epochs..." % n_epochs)

    model.fit(X, Y, batch_size=64, epochs=n_epochs, validation_split=.1)

    model.save(fname, overwrite=True)
