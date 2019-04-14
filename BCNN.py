import numpy as np
import tensorflow as tf

from Model import Model
from JFnet import JFnet

from tensorflow.keras.layers import Conv2D, Dropout, Dense, MaxPool2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Input, LeakyReLU, Softmax, Reshape, Flatten
from tensorflow.keras.layers import concatenate, maximum, Lambda, Layer
from tensorflow.keras import regularizers
import pickle


class BCNN(Model):
    """Bayesian convolutional neural network (if p != 0 and on at test time)
    Save weights are for last_layer='17', but default was 13
    Weights:
    models/weights_bcnn1_392bea6.h5
    models/weights_bcnn2_b69aadd.h5
    """

    def __init__(self, p_conv=0.2, last_layer="layer_17", weights=None,
                 n_classes=2, l1_lambda=0.001, **kwargs):
        """
        weights - path to the weight .h5 file
        Regularization type hardcoded (l1)
        """
        jf_model = JFnet.build_model(
            width=512, height=512,
            filename=JFnet.WEIGHTS_PATH,
            p_conv=p_conv, **kwargs)
        # remove unused layers
        conv_output = jf_model.get_layer(last_layer).output

        mean_pooled = GlobalAveragePooling2D(
            data_format='channels_last')(conv_output)
        max_pooled = GlobalMaxPooling2D(
            data_format='channels_last')(conv_output)
        global_pool = concatenate([mean_pooled, max_pooled], axis=1)

        softmax_input = Dense(
            units=n_classes,
            activation=None,
            kernel_regularizer=regularizers.l1(l1_lambda)
            )(global_pool)
        softmax_output = Softmax()(softmax_input)

        model = tf.keras.Model(
            inputs=[jf_model.input[0]],
            outputs=[softmax_output])

        if weights is not None:
            model.load_weights(weights)

        super(BCNN, self).__init__(net=model)


# testing with main
if __name__ == "__main__":
    # avoid using batchsize with BCNN
    bcnn = BCNN(weights="models/weights_bcnn1_392bea6.h5")
    bcnn.print_summary()
    model = bcnn.net
    input = np.zeros(model.input_shape[1:])
    input = input.reshape((-1,) + model.input_shape[1:])
    print("-" * 10 + "Predict" + "-" * 10)
    print(bcnn.predict(input))
    print("-" * 10 + "MC samples" + "-" * 10)
    print(bcnn.mc_samples(input))
