"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, GaussianNoise, Activation,Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.layers.merge import Concatenate, Add
import keras.backend as K


def conv_block2(n_filter, n1, n2,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs
                ):
    def _func(lay):
        s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init,**kwargs)(lay)
        if batch_norm:
            s = BatchNormalization()(s)
        s = Activation(activation)(s)
        if dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func



def unet_block(n_depth=2, n_filter_base=16, n_row=3, n_col=3, n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None):
    """"""

    if last_activation is None:
        last_activation = activation

    if K.image_dim_ordering() == "tf":
        channel_axis = -1
    else:
        channel_axis = 1


    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block2(n_filter_base * 2 ** n, n_row, n_col,
                                    dropout=dropout,
                                    activation=activation,
                                    batch_norm=batch_norm, name = "down_level_%s_no_%s"%(n,i))(layer)
            skip_layers.append(layer)
            layer = MaxPooling2D((2, 2), name = "max_%s"%n)(layer)


        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block2(n_filter_base * 2 ** n_depth, n_row, n_col,
                                dropout=dropout,
                                activation=activation,
                                batch_norm=batch_norm,name = "middle_%s"%i)(layer)

        layer = conv_block2(n_filter_base * 2 ** (n_depth - 1), n_row, n_col,
                            dropout=dropout,
                            activation=activation,
                            batch_norm=batch_norm,name = "middle_%s"%(n_conv_per_depth-1))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis = channel_axis)([UpSampling2D((2, 2))(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block2(n_filter_base * 2 ** n, n_row, n_col,
                                    dropout=dropout,
                                    activation=activation,
                                    batch_norm=batch_norm,name = "up_level_%s_no_%s"%(n,i))(layer)

            layer = conv_block2(n_filter_base * 2 ** max(0, n - 1), n_row, n_col,
                                dropout=dropout,
                                activation=activation if n > 0 else last_activation,
                                batch_norm=batch_norm, name = "up_level_%s_no_%s"%(n,n_conv_per_depth-1))(layer)

        return layer

    return _func



def resunet_model(input_shape, last_activation, n_depth=2, n_filter_base=16, n_row=3, n_col=3, n_conv_per_depth=2,
                  activation="relu",
                  batch_norm=False,
                  n_channel_out = None,
                  dropout=0.0):
    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid'. 'relu')!")



    if n_channel_out is None:
        if K.image_dim_ordering() == "tf":
            n_channel_out = input_shape[-1]
        else:
            n_channel_out = input_shape[0]

    input = Input(input_shape)
    unet = unet_block(n_depth, n_filter_base, n_row, n_col,
                      activation=activation,
                      dropout=dropout,
                      n_conv_per_depth = n_conv_per_depth)(input)

    final = Add()([Conv2D(n_channel_out, (3, 3), activation='linear', padding = "same", name = "final_residual")(unet), input])
    final = Activation(activation=last_activation)(final)

    return Model(inputs=input, outputs=final)

