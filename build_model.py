import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Add, Concatenate, Conv3D, Dense, Dropout, Flatten, Input, Lambda, LeakyReLU, Multiply, Reshape
from keras.models import Model


### Utils ###
def k_mean(tensor):
    return K.mean(tensor, axis=2)


def k_sqrt(tensor):
    r = K.sqrt(tensor)
    return r


def k_atan(tensor):
    import tensorflow as tf

    t = tf.math.atan2(tensor[0], tensor[1])
    return t


def ifft(x):
    y = tf.complex(x[:, :, :, 0], x[:, :, :, 1])
    ifft = tf.signal.ifft(y)
    return tf.stack([tf.math.real(ifft), tf.math.imag(ifft)], axis=3)


### Model definition ###
def build_model():
    # Model specifications
    dropout_rate = 0.25
    num_complex_channels = 6
    nn_input = Input((64, 100, 2))

    mean_input = Lambda(k_mean)(nn_input)
    print(mean_input.get_shape())

    # complex to polar
    real = Lambda(lambda x: x[:, :, :, 0])(nn_input)
    imag = Lambda(lambda x: x[:, :, :, 1])(nn_input)

    real_squared = Multiply()([real, real])
    imag_squared = Multiply()([imag, imag])

    real_imag_squared_sum = Add()([real_squared, imag_squared])

    # amplitude
    r = Lambda(k_sqrt)(real_imag_squared_sum)
    r = Reshape((64, 100, 1))(r)
    print(r.get_shape())

    # phase
    t = Lambda(k_atan)([imag, real])
    t = Reshape((64, 100, 1))(t)
    print(t.get_shape())

    polar_input = Concatenate()([r, t])

    time_input = Lambda(ifft)(nn_input)

    total_input = Concatenate()([nn_input, polar_input, time_input])
    print("total", total_input.get_shape())

    # reduce dimension of time axis
    lay_input = Reshape((64, 100, num_complex_channels, 1))(total_input)

    layD1 = Conv3D(8, (1, 23, num_complex_channels), strides=(1, 5, 1), padding="same")(lay_input)
    layD1 = LeakyReLU(alpha=0.3)(layD1)
    layD1 = Dropout(dropout_rate)(layD1)
    layD2 = Conv3D(8, (1, 23, 1), padding="same")(layD1)
    layD2 = LeakyReLU(alpha=0.3)(layD2)
    layD2 = Concatenate()([layD1, layD2])
    layD2 = Conv3D(8, (1, 1, num_complex_channels), padding="same")(layD2)
    layD2 = LeakyReLU(alpha=0.3)(layD2)
    layD2 = Conv3D(8, (1, 23, 1), strides=(1, 5, 1), padding="same", kernel_regularizer=regularizers.l2(0.01))(layD2)
    layD2 = LeakyReLU(alpha=0.3)(layD2)
    layD2 = Dropout(dropout_rate)(layD2)
    layD3 = Conv3D(8, (1, 23, 1), padding="same")(layD2)
    layD3 = LeakyReLU(alpha=0.3)(layD3)
    layD3 = Concatenate()([layD2, layD3])
    layD3 = Conv3D(8, (1, 1, num_complex_channels), padding="same")(layD3)
    layD3 = LeakyReLU(alpha=0.3)(layD3)
    layD3 = Conv3D(8, (1, 23, 1), strides=(1, 5, 1), padding="same", kernel_regularizer=regularizers.l2(0.01))(layD3)
    layD3 = LeakyReLU(alpha=0.3)(layD3)
    layD3 = Dropout(dropout_rate)(layD3)
    layD4 = Conv3D(8, (1, 23, 1), padding="same")(layD3)
    layD4 = LeakyReLU(alpha=0.3)(layD4)
    layD4 = Concatenate()([layD4, layD3])
    layD4 = Conv3D(8, (1, 1, num_complex_channels), padding="same")(layD4)
    layD4 = LeakyReLU(alpha=0.3)(layD4)

    layV1 = Conv3D(8, (8, 1, 1), padding="same")(layD4)
    layV1 = LeakyReLU(alpha=0.3)(layV1)
    layV1 = Dropout(dropout_rate)(layV1)
    layV1 = Concatenate()([layV1, layD4])
    layV2 = Conv3D(8, (8, 1, 1), padding="same", kernel_regularizer=regularizers.l2(0.01))(layV1)
    layV2 = LeakyReLU(alpha=0.3)(layV2)
    layV2 = Dropout(dropout_rate)(layV2)
    layV2 = Concatenate()([layV2, layV1])
    layV3 = Conv3D(8, (8, 1, 1), padding="same")(layV2)
    layV3 = LeakyReLU(alpha=0.3)(layV3)
    layV3 = Dropout(dropout_rate)(layV3)
    layV3 = Concatenate()([layV3, layV2])
    layV4 = Conv3D(8, (8, 1, 1), padding="same")(layV3)
    layV4 = LeakyReLU(alpha=0.3)(layV4)
    layV4 = Dropout(dropout_rate)(layV4)
    layV4 = Concatenate()([layV4, layV3])
    layV5 = Conv3D(8, (8, 1, 1), padding="same")(layV4)
    layV5 = LeakyReLU(alpha=0.3)(layV5)
    layV5 = Dropout(dropout_rate)(layV5)

    nn_output = Flatten()(layV5)
    nn_output = Dense(64, activation="relu")(nn_output)
    nn_output = Dense(32, activation="relu")(nn_output)
    nn_output = Dense(2, activation="linear")(nn_output)
    model = Model(inputs=nn_input, outputs=nn_output)
    model.compile(optimizer="Adam", loss="mse")
    model.summary()
    return model
