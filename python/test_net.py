"""
Test the network on some new noisy images

The network architecture and weights are stored in 
 
 models/resunet.hdf5
 
 and loaded with the keras function 'load_model'
 
This should probably have a deeplearning4j equivalent
 
Further, the input data (images) have to be cast into the 4D shape
 
 (n_samples, n_height, n_width, n_channels)
 


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


    # create input/output pairs
    print("creating samples...")
    X, Y = create_words_iter(("PURE", "VERNUNFT", "DARF", "NIEMALS", "SIEGEN"),
                             sigma_noise=1.9,
                             size = (300,72),
                             pos=(15,5))

    # load the model
    print("building model...")
    model = load_model(fname)

    # apply model
    print("predicting...")
    Y_pred = model.predict(X)

    # plot the result
    plt.figure(facecolor = "k")

    for i, (U, title) in enumerate(zip((X, Y_pred, Y), ("input", "network", "original"))):
        for j, u in enumerate(U):
            plt.subplot(3, len(X), i * len(X) + j + 1)
            plt.imshow(np.squeeze(u), cmap="gray")
            plt.axis("off")

            if j == len(U) // 2:
                plt.title(title, color = "w")

    plt.show()
