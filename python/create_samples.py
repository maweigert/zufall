"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import keras.backend as K
from PIL import Image, ImageDraw, ImageFont

TEXT_IMAGE_SIZE = (320, 96)


def create_sample_stripes(n_samples, n_size=128, sigma_noise=.9):
    # create some random stripe patterns
    x0 = np.arange(n_size)
    _Y0, _X0 = np.meshgrid(x0, x0, indexing="ij")
    angles = np.random.normal(0, 1, (n_samples, 2, 1, 1))
    angles *= 1. / np.linalg.norm(angles, axis=1, keepdims=True)

    _X1 = _X0 + .6 / n_size * np.random.uniform(-1, 1, (n_samples, 1, 1)) * _X0 ** 2
    _A = angles[:, 0] * _X1 + angles[:, 1] * _Y0

    Y = .5 + .5 * np.sin(_A)[:, np.newaxis, :, :]
    X = Y + sigma_noise * np.random.normal(0, 1, Y.shape)

    # normalize
    mi = np.percentile(X, 1, axis=(1, 2, 3), keepdims=True)
    ma = np.percentile(X, 99, axis=(1, 2, 3), keepdims=True)
    X = (1. * X - mi) / (ma - mi)

    if K.image_dim_ordering() == "tf":
        X = np.swapaxes(X, 1, -1)
        Y = np.swapaxes(Y, 1, -1)

    return X, Y


def _image_from_word(word, pos=(0, 0), size=None):
    if size is None:
        size = TEXT_IMAGE_SIZE
    #print("create image for text = '%s ...'" % word)
    im = Image.new("RGB", size)
    fnt = ImageFont.truetype("data/xkcd.ttf", size=70)
    ImageDraw.Draw(im).text(pos, word, font=fnt)
    return 1 / 255. * np.asarray(im)[..., 0].T

def forward_model(y, sigma_noise=1.9):
    x = y + sigma_noise * np.random.normal(0, 1, y.shape)
    return x
    

def create_words_iter(words, sigma_noise=1.9, size  =None, pos=None, add_exclamation = True):
    excla = ["!",":","?"]
    def _single(word):
        if np.random.randint(0,2)==0:
            word = word + excla[np.random.randint(0,len(excla))]
        if pos is None:
            x = TEXT_IMAGE_SIZE[0] * np.random.uniform(0.1, .3)
            y = TEXT_IMAGE_SIZE[1] * np.random.uniform(0.1, .3)
            return _image_from_word(word, pos=(int(x), int(y)), size=size)
        else:
            return _image_from_word(word, pos=pos, size=size)

    Y = np.stack([_single(w) for w in words])
    Y = Y[:, np.newaxis]
    X = forward_model(Y,sigma_noise)

    # normalize
    mi = np.percentile(X, 1, axis=(1, 2, 3), keepdims=True)
    ma = np.percentile(X, 99, axis=(1, 2, 3), keepdims=True)
    X = (1. * X - mi) / (ma - mi)

    if K.image_dim_ordering() == "tf":
        X = np.swapaxes(X, 1, -1)
        Y = np.swapaxes(Y, 1, -1)

    return X, Y


def create_sample_words(n_samples, sigma_noise=.9):
    a = np.genfromtxt("data/words.txt", dtype=str)
    np.random.shuffle(a)
    return create_words_iter(a[:n_samples], sigma_noise=sigma_noise)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X, Y = create_sample_words(10)

    plt.figure(facecolor = "k")

    for i, (U, title) in enumerate(zip((X[:3], Y[:3]), ("input (X)", "original (Y)"))):
        for j, u in enumerate(U):
            plt.subplot(2, len(X[:3]), i * len(X[:3]) + j + 1)
            plt.imshow(np.squeeze(u), cmap="gray")
            plt.axis("off")

            if j == len(U) // 2:
                plt.title(title, color = "w")

    plt.show()
