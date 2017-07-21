"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from create_samples import create_samples

import matplotlib.pyplot as plt
from train_net import build_model

if __name__ == '__main__':
    fname = "models/resunet.hdf5"

    print("creating samples...")
    X, Y = create_samples(3)

    print("building model...")
    model = build_model(X.shape[1:])
    model.load_weights(fname)

    print("predicting...")
    Y_pred = model.predict(X)

    plt.figure()

    for i, U in enumerate((X, Y, Y_pred)):
        for j, u in enumerate(U):
            plt.subplot(3, len(X), i * 3 + j + 1)
            plt.imshow(np.squeeze(u))
            plt.axis("off")

    plt.show()
