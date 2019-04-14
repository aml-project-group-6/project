import numpy as np
import tensorflow as tf

from Model import Model

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, Dense, MaxPool2D
from tensorflow.keras.layers import Input, LeakyReLU, Softmax, Reshape, Flatten
from tensorflow.keras.layers import concatenate, Lambda, Layer
from tensorflow.keras import regularizers

class JFnet(Model):

    WEIGHTS_PATH = "models/keras_JFnet.h5"

    def __init__(self, width=512, height=512, **kwargs):
        network = JFnet.build_model(width=width, height=height,
                                    filename=JFnet.WEIGHTS_PATH,
                                    **kwargs
                                    )
        super(JFnet, self).__init__(net=network)

    @staticmethod
    def build_model(width=512, height=512, filename=None,
                    n_classes=5, batch_size=None, p_conv=0.0,
                    l2_lambda=0.001
                    ):
        """
        Provide batch size for JFNET
        Keep it None for BCNN, easier to deal with inputs
        Regularization type hardcoded (l2)
        """
        # Input shape (height, width, depth)
        # different from original implimentation!
        main_input = Input(
            shape=(height, width, 3),
            batch_size=batch_size,
        )

        # Note: for conv layers paper uses untie_biases=True
        # layer will have separate bias parameters for entire output
        # As a result, the bias is a 3D tensor.
        # Implemented as a Bias layer

        # no need to init weights as they will be loaded from a file
        # Conv layers(filters, kernel_size)
        conv_main_1 = Conv2D(
            32, 7, strides=(2, 2), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(main_input)
        conv_bias_1 = Bias()(conv_main_1)
        conv_activation_1 = LeakyReLU(alpha=0.5)(conv_bias_1)
        dropout_1 = Dropout(rate=p_conv)(conv_activation_1)
        maxpool_1 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_1)
        # 3
        conv_main_2 = Conv2D(
            32, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(maxpool_1)
        conv_bias_2 = Bias()(conv_main_2)
        conv_activation_2 = LeakyReLU(alpha=0.5)(conv_bias_2)
        dropout_2 = Dropout(rate=p_conv)(conv_activation_2)
        # 4
        conv_main_3 = Conv2D(
            32, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_2)
        conv_bias_3 = Bias()(conv_main_3)
        conv_activation_3 = LeakyReLU(alpha=0.5)(conv_bias_3)
        dropout_3 = Dropout(rate=p_conv)(conv_activation_3)
        maxpool_3 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_3)
        # 6
        conv_main_4 = Conv2D(
            64, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(maxpool_3)
        conv_bias_4 = Bias()(conv_main_4)
        conv_activation_4 = LeakyReLU(alpha=0.5)(conv_bias_4)
        dropout_4 = Dropout(rate=p_conv)(conv_activation_4)
        # 7
        conv_main_5 = Conv2D(
            64, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_4)
        conv_bias_5 = Bias()(conv_main_5)
        conv_activation_5 = LeakyReLU(alpha=0.5)(conv_bias_5)
        dropout_5 = Dropout(rate=p_conv)(conv_activation_5)
        maxpool_5 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_5)
        # 9
        conv_main_6 = Conv2D(
            128, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(maxpool_5)
        conv_bias_6 = Bias()(conv_main_6)
        conv_activation_6 = LeakyReLU(alpha=0.5)(conv_bias_6)
        dropout_6 = Dropout(rate=p_conv)(conv_activation_6)
        # 10
        conv_main_7 = Conv2D(
            128, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_6)
        conv_bias_7 = Bias()(conv_main_7)
        conv_activation_7 = LeakyReLU(alpha=0.5)(conv_bias_7)
        dropout_7 = Dropout(rate=p_conv)(conv_activation_7)
        # 11
        conv_main_8 = Conv2D(
            128, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_7)
        conv_bias_8 = Bias()(conv_main_8)
        conv_activation_8 = LeakyReLU(alpha=0.5)(conv_bias_8)
        dropout_8 = Dropout(rate=p_conv)(conv_activation_8)
        # 12
        conv_main_9 = Conv2D(
            128, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_8)
        conv_bias_9 = Bias()(conv_main_9)
        conv_activation_9 = LeakyReLU(alpha=0.5)(conv_bias_9)
        dropout_9 = Dropout(rate=p_conv)(conv_activation_9)
        maxpool_9 = MaxPool2D(
            pool_size=3, name="layer_13",
            strides=(2, 2))(dropout_9)
        # 14
        conv_main_10 = Conv2D(
            256, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(maxpool_9)
        conv_bias_10 = Bias()(conv_main_10)
        conv_activation_10 = LeakyReLU(alpha=0.5)(conv_bias_10)
        dropout_10 = Dropout(rate=p_conv)(conv_activation_10)
        # 15
        conv_main_11 = Conv2D(
            256, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_10)
        conv_bias_11 = Bias()(conv_main_11)
        conv_activation_11 = LeakyReLU(alpha=0.5)(conv_bias_11)
        dropout_11 = Dropout(rate=p_conv)(conv_activation_11)
        # 16
        conv_main_12 = Conv2D(
            256, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_11)
        conv_bias_12 = Bias()(conv_main_12)
        conv_activation_12 = LeakyReLU(alpha=0.5)(conv_bias_12)
        dropout_12 = Dropout(rate=p_conv)(conv_activation_12)
        # 17
        conv_main_13 = Conv2D(
            256, 3, strides=(1, 1), padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(dropout_12)
        conv_bias_13 = Bias()(conv_main_13)
        conv_activation_13 = LeakyReLU(alpha=0.5, name="layer_17")(conv_bias_13)
        dropout_13 = Dropout(rate=p_conv, name="layer_17d")(conv_activation_13)
        maxpool_13 = MaxPool2D(
            pool_size=3, strides=(2, 2),
            name="last_conv",
            )(dropout_13)
        # 19, special dropout between phases with p=1/2
        dropout_inter = Dropout(rate=0.5)(maxpool_13)
        flatten_inter = Flatten()(dropout_inter)
        # 20 Dense phase
        # Maxout layer is implemented here as Dense+custom feature_pool
        maxout_1 = Dense(units=1024,
                         activation=None,)(flatten_inter)
        # need to wrap operation in Lambda to count as a layer
        maxout_2 = Lambda(
            lambda x: feature_pool_max(x, pool_size=2, axis=1)
            )(maxout_1)

        # 22 Concatenate with processed img, take both eyes into account
        img_dim_input = Input(shape=(2,), batch_size=batch_size, name="imgdim")
        concat = concatenate([maxout_2, img_dim_input], axis=1)

        # 24
        # use lambda for custom reshape
        # that's capable of changing batch_size as well
        # expect order left-right
        # TODO: (-1, net['23'].output_shape[1] * 2)
        flatten = Lambda(
            lambda x: tf.reshape(x, (-1, concat.shape[1]*2))
        )(concat)
        dense_dropout_0 = Dropout(rate=0.5)(flatten)
        # 26
        dense_1 = Dense(units=1024,
                        activation=None,
                        )(dense_dropout_0)
        dense_maxpool_1 = Lambda(
            lambda x: feature_pool_max(x, pool_size=2, axis=1)
            )(dense_1)
        dense_dropout_1 = Dropout(rate=0.5)(dense_maxpool_1)

        # 29
        dense_2 = Dense(units=n_classes*2,
                        activation=None,)(dense_dropout_1)
        softmax_flatten = Lambda(
            lambda x: tf.reshape(x, (-1, n_classes))
            )(dense_2)
        softmax = Softmax()(softmax_flatten)

        model = tf.keras.Model(
            inputs=[main_input, img_dim_input],
            outputs=[softmax]
        )

        if filename:
            model.load_weights(filename)

        return model

    @staticmethod
    def get_img_dim(width, height):
        """Second input to JFnet consumes image dimensions
        division by 700 according to https://github.com/JeffreyDF/
        kaggle_diabetic_retinopathy/blob/
        43e7f51d5f3b2e240516678894409332bb3767a8/generators.py::lines 41-42
        """
        return np.vstack((width, height)).T / 700.


class Bias(Layer):
    """
    Adds bias to a layer. This is used for untied biases convolution.
    """
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer='uniform',
            trainable=True)
        super(Bias, self).build(input_shape)

    def call(self, x):
        return tf.add(x, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


def feature_pool_max(input, pool_size=2, axis=1):
    """
    Based on lasagne implementation of FeaturePool
    """
    input_shape = input.shape.as_list()
    num_feature_maps = input_shape[axis]
    num_feature_maps_out = num_feature_maps // pool_size

    pool_shape = tf.TensorShape(
        (input_shape[1:axis] + [num_feature_maps_out, pool_size] + input_shape[axis+1:])
    )

    input_reshaped = Reshape(pool_shape)(input)
    # reduce along all axis but the target one
    reduction_axis = list(range(1, len(pool_shape)+1))
    reduction_axis.pop(axis-1)

    return tf.reduce_max(input_reshaped, axis=reduction_axis)


# testing with main
if __name__ == "__main__":
    """
    model = JFnet.build_model()
    print(model.summary())
    """
    # Use batchsize with JFNET
    jfnet = JFnet(batch_size=64)
    jfnet.print_summary()
    model = jfnet.net
    input = [np.zeros(model.input_shape[0]), np.zeros(model.input_shape[1])]
    print("-" * 10 + "Predict" + "-" * 10)
    print(jfnet.predict(input))
    print("-" * 10 + "MC samples" + "-" * 10)
    print(jfnet.mc_samples(input, n_inputs=2))
