import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Progbar

if __name__ == '__main__':
    import os
    os.sys.path.append('.')
from datasets import KaggleDR, Messidor

from BCNN import BCNN
from JFnet import JFnet

# Configuration
def get_monte_carlo_sample_count():
    return 100
def get_bcnn_weights_filepath(weights_type):
    return '../training_output/bcnn{}.h5'.format(weights_type)
def get_bcnn_last_layer_name():
    return 'layer_17d'
def get_bcnn_dropout_probability():
    return 0.2
def get_bcnn_class_count():
    return 2
def get_batch_size():
    return 64

def get_dataset_images_path():
    return '../../messidor/'
def get_dataset_labels_path():
    return '../../messidor/messidor.csv'

def get_prediction_output_path(model_name, weights_type):
    return '../predict_output/mc_100_messidor_{}_{}.pkl'.format(model_name.lower(), weights_type)

def load_model(model_name, weights_type):
    if model_name == 'BCNN':
        return BCNN(
            p_conv=get_bcnn_dropout_probability(),
            last_layer=get_bcnn_last_layer_name(),
            n_classes=get_bcnn_class_count(),
            weights=get_bcnn_weights_filepath(weights_type))
    if model_name == 'JFNet':
        return JFnet(batch_size=get_batch_size())
    raise Exception('Unhandled model')

def get_model_input_count(model_name):
    if model_name == 'BCNN': return 1
    if model_name == 'JFNet': return 2
    raise Exception('Unhandled model')

def prepare_input(model_name, X):
    if model_name == 'JFNet':
        img_dim = np.zeros((X.shape[0], 2))
        img_dim[:, 0] = 512 # width
        img_dim[:, 1] = 512 # height
        return [X, img_dim]
    if model_name == 'BCNN':
        return X
    raise Exception('Unhandled model')

# Main entry point.
def main(model_name, weights_type):
    # Load the model.
    model = load_model(model_name, weights_type)
    input_count = get_model_input_count(model_name)

    # Load the dataset labels.
    labels = pd.read_csv(get_dataset_labels_path())
    labels.image = labels.image.apply(lambda s: s + '.jpeg')
    labels.level = labels.level.astype(str)
    sample_count = labels.shape[0]

    # data is kept in folders with images with correspdonding csv file with labelss
    keras_img_gen = ImageDataGenerator(preprocessing_function=KaggleDR.standard_normalize)
    generator = keras_img_gen.flow_from_dataframe(
        labels,
        directory=get_dataset_images_path(),
        x_col='image',
        y_col='level',
        target_size=(512, 512),
        batch_size=get_batch_size(),
        shuffle=False,
        seed=None)

    out_dim = model.net.output_shape[1]
    det_out = np.zeros((sample_count, out_dim), dtype=np.float32)
    stoch_out = np.zeros((sample_count, out_dim, get_monte_carlo_sample_count()), dtype=np.float32)

    index = 0
    progbar = Progbar(sample_count)
    for X, y in generator:
        window_size = X.shape[0]
        if window_size == 0:
            break
        if index >= det_out.shape[0]:
            break
        inputs = prepare_input(model_name, X)
        det_out[index:index + window_size] = model.predict(inputs)
        stoch_out[index:index + window_size] = model.mc_samples(
            inputs,
            n_inputs=input_count,
            T=get_monte_carlo_sample_count())
        index += window_size
        progbar.add(window_size)
        if index == sample_count:# or (index == 64*(sample_count//64) and model_name == 'JFNet'):
            break

    # Write prediction results.
    with open(get_prediction_output_path(model_name, weights_type), 'wb') as f:
        pickle.dump({'det_out': det_out, 'stoch_out': stoch_out}, f)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print('  python messidor_predict.py (BCNN|JFNet) (0vs1234|01vs234)')
        exit()
    model_name = sys.argv[1]
    weights_type = sys.argv[2]
    main(model_name, weights_type)
