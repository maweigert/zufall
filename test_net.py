"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from create_samples import create_words_iter
from keras.models import load_model

import matplotlib.pyplot as plt
from train_net import build_model

if __name__ == '__main__':
    fname = "models/resunet.hdf5"

    print("creating samples...")
    X, Y = create_words_iter(("PURE", "VERNUNFT", "DARF", "NIEMALS", "SIEGEN"),
                             sigma_noise=1.9,
                             size = (300,72),
                             pos=(15,5))

    print("building model...")
    model = load_model(fname)

    print("predicting...")
    Y_pred = model.predict(X)

    plt.figure(facecolor = "k")

    for i, (U, title) in enumerate(zip((X, Y_pred, Y), ("input", "network", "original"))):
        for j, u in enumerate(U):
            plt.subplot(3, len(X), i * len(X) + j + 1)
            plt.imshow(np.squeeze(u), cmap="gray")
            plt.axis("off")

            if j == len(U) // 2:
                plt.title(title, color = "w")

    plt.show()
