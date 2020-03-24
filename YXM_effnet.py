from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=0)
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=0)
    return SSR/SST

def get_post(x_in):
    x = LeakyReLU()(x_in)
    x = BatchNormalization()(x)
    return x

def get_block(x_in, ch_in, ch_out, filter):
    '''

    Args:
        x_in: 输入大小
        ch_in: 输入通道数量
        ch_out: 输出通道数量

    Returns: 输出大小为

    '''
    x = Conv2D(ch_in,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False)(x_in)
    x = get_post(x)

    x = DepthwiseConv2D(kernel_size=(1, filter), padding='same', use_bias=False)(x)
    x = get_post(x)
    x = MaxPool2D(pool_size=(2, 1),
                  strides=(2, 1))(x) # Separable pooling

    x = DepthwiseConv2D(kernel_size=(filter, 1),
                        padding='same',
                        use_bias=False)(x)
    x = get_post(x)
    x = Conv2D(ch_out,
               kernel_size=(2, 1),
               strides=(1, 2),
               padding='same',
               use_bias=False)(x)
    x = get_post(x)

    return x


def Effnet(input_shape, nb_classes = 0, classification_tesk=False, weights=None, regression_tesk = False):
    x_in = Input(shape=input_shape)

    x = get_block(x_in, 8, 8, 5)
    x = get_block(x, 8, 8, 5)
    x = get_block(x, 8, 16, 4)
    x = get_block(x, 16, 16, 5)
    x = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    if classification_tesk:
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)
    if regression_tesk:
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        x = Dense(1, activation='relu')(x)


    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)
    # print(model.summary())

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_square])
    return model

if __name__ == '__main__':
    shape = (128, 128, 3)
    Effnet(shape, regression_tesk=True)