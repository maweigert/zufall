"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import keras.backend as K
from PIL import Image, ImageDraw, ImageFont



def create_stripes(n_samples, n_size = 128, sigma_noise = .9):


    # create some random stripe patterns
    x0 = np.arange(n_size)
    _Y0,_X0 = np.meshgrid(x0,x0, indexing = "ij")
    angles = np.random.normal(0, 1, (n_samples,2,1,1))
    angles *= 1./np.linalg.norm(angles, axis = 1, keepdims = True)


    _X1 = _X0+.6/n_size*np.random.uniform(-1, 1, (n_samples, 1, 1)) * _X0 ** 2
    _A = angles[:, 0] * _X1 + angles[:, 1] * _Y0

    Y = .5+.5*np.sin(_A)[:, np.newaxis,:,:]
    X = Y + sigma_noise * np.random.normal(-1, 1, Y.shape)

    if K.image_dim_ordering() == "tf":
        X = np.swapaxes(X, 1, -1)
        Y = np.swapaxes(Y, 1, -1)

    return X, Y





def create_letters(n_samples, n_size = 128, sigma_noise = .9):
    def _single(text):
        print("create image for text = '%s ...'"%text)
        im = Image.new("RGB", (n_size, n_size))
        fnt = ImageFont.truetype("Arial", size=30*n_size//128)
        pos = n_size*np.random.uniform(0,.5,2)

        ImageDraw.Draw(im).text(pos.astype(int), text, font=fnt)
        return 1/255.*np.asarray(im)[...,0].T

    Y = np.stack([_single("".join(chr(np.random.randint(65,91)) for _ in range(6))) for _ in range(n_samples)])
    Y = Y[:,np.newaxis]
    X = Y + sigma_noise * np.random.normal(-1, 1, Y.shape)

    if K.image_dim_ordering() == "tf":
        X = np.swapaxes(X, 1, -1)
        Y = np.swapaxes(Y, 1, -1)

    return X, Y



if __name__ == '__main__':

    X,Y = create_letters(10)