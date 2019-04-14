import numpy as np
import tensorflow.keras.backend as K


class Model():
    """
    Base class for our Keras models
    predict is Keras.Model.predict
    net is of type Keras.Model
    """
    def __init__(self, net=None):
        self.net = net
        self._predict = None
        self._predict_stoch = None

    def predict(self, *inputs):
        """Forward pass"""
        # prediction corresponds to keras.Model.predict
        if self._predict is None:
            self._predict = self.net.predict
        return self._predict(*inputs)

    def mc_samples(self, model_input, *inputs, T=100, n_inputs=1):
        """
        Stochastic forward passes to generate T MC samples
        Assumes inputs to the model is the first argument!
        Currently works only with 1 input model
        """
        # Implement stochastic prediction
        # keep dropout layers by setting learning phase to True
        # create function that accepts training mode as parameter
        # all inputs of form [input, 1]
        if self._predict_stoch is None:
            self._predict_stoch = K.function(
                [self.net.input, K.learning_phase()],
                self.net.output,
            )
        # n_samples - batch size in this context
        if n_inputs == 1:
            n_samples = len(model_input)
        else:
            n_samples = len(model_input[0])

        n_out = self.net.layers[-1].output_shape[1]
        mc_samples = np.zeros((n_samples, n_out, T))
        # need to add 1 to inputs for learning phase
        for t in range(T):
            mc_samples[:, :, t] = self._predict_stoch([model_input, 1], *inputs)
        return mc_samples

    def get_output_layer(self):
        return self.net.layers[-1]

    # print model summary
    def print_summary(self):
        if self.net is None:
            print("Model undefined")
        else:
            self.net.summary()
