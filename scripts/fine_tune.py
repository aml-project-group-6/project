# Run from root directory
import gc
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    import os
    os.sys.path.append('.')

from BCNN import BCNN
from datasets import KaggleDR
from datasets import DatasetImageDataGenerator
# from training import generator_queue
# from util import Progplot

# --------- Define parameters ---------
p = 0.2
last_layer = 'layer_17d'  # from JFnet
batch_size = 32
epochs = 20
# if want to just test performance set train_model to false
train_model = False
# lr_schedule = {0: 0.005, 1: 0.005, 2: 0.001, 3: 0.001, 4: 0.0005, 5: 0.0001}
# change_every = 5
# we don't apply regularization to bias
l2_lambda = 0.001  # entire network
l1_lambda = 0.001  # only last layer
size = 512
n_classes = 2
# lr_schedule={0: 0.005, 1: 0.005, 2: 0.001, 3: 0.001, 4: 0.0005, 5: 0.0001}
seed = 1234

train_dir = "../../output/"
test_dir = "../../output_test/"
save_dir = "../training_output/"
model_name = "bcnn01vs234"

# None to have new model, without pretraining
weights_path = "../training_output/bcnn01vs234.h5"
# paper weights

# --------- Dataset creation ---------
# parameters for augmenting data
# Currently need to specify preprocessing function manually
AUGMENTATION_PARAMS = {'featurewise_center': False,
                       'samplewise_center': False,
                       'featurewise_std_normalization': False,
                       'samplewise_std_normalization': False,
                       'zca_whitening': False,
                       'rotation_range': 180.,
                       'width_shift_range': 0.05,
                       'height_shift_range': 0.05,
                       'shear_range': 0.,
                       'zoom_range': 0.10,
                       'channel_shift_range': 0.,
                       'fill_mode': 'constant',
                       'cval': 0.,
                       'horizontal_flip': True,
                       'vertical_flip': True,
                       #'dim_ordering': 'th'
                       'data_format' : 'channels_last',
                       # Preprocessing function ONLY FOR KAGGLE DR
                       'preprocessing_function' : KaggleDR.standard_normalize,
                       'validation_split':0.2,
                      }

train_datagen = ImageDataGenerator(**AUGMENTATION_PARAMS)

labels = pd.read_csv(train_dir + "trainLabels01vs234.csv")

labels['image'] = labels['image'].apply(lambda x: x + '.jpeg')
labels['level'] = labels['level'].astype(str)

# same preprocessing for test labels
test_labels = pd.read_csv(test_dir + "testLabels01vs234.csv")
test_labels['image'] = test_labels['image'].apply(lambda x: x + '.jpeg')
test_labels['level'] = test_labels['level'].astype(str)

# create dataset using folder directories
# train feeds into augmenter
train_generator = train_datagen.flow_from_dataframe(
    labels,
    directory=train_dir,
    x_col='image',
    y_col='level',
    target_size=(512, 512),
    batch_size=batch_size,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    subset='training'
)

# dataset for validation score
validation_generator = train_datagen.flow_from_dataframe(
    labels,
    directory=train_dir,
    x_col='image',
    y_col='level',
    target_size=(512, 512),
    batch_size=batch_size,
    shuffle=False,
    seed=None,
    save_to_dir=None,
    subset='validation'
)

# test set
test_datagen = ImageDataGenerator(
    preprocessing_function=KaggleDR.standard_normalize,
)

test_generator = test_datagen.flow_from_dataframe(
    test_labels,
    directory=test_dir,
    x_col='image',
    y_col='level',
    target_size=(512, 512),
    batch_size=batch_size,
    shuffle=False,
    seed=None,
)

# --------- Compiling Model ---------
# Setup networks
bcnn = BCNN(p_conv=p, last_layer=last_layer, n_classes=n_classes,
            l1_lambda=l1_lambda, l2_lambda=l2_lambda,
            weights=weights_path
           )
model = bcnn.net

# TODO: make this work
def bayes_cross_entropy(y, ce_loss, n_classes):
    """Dalyac et al. (2014), eq. (17)"""
    # changed bincount to use numpy
    # need to shape to 1D from (len, 1)
    y = tf.cast(y, dtype=tf.int32)
    priors = tf.bincount(y) / y.shape[1]
    weights = 1.0 / (priors[y] * y.shape[1] * n_classes)
    bce_loss = ce_loss * weights
    return bce_loss.sum()


# custom loss function for the model
def bce_loss(n_classes):
    ce_loss = CategoricalCrossentropy()

    def loss(y_true, y_pred):
        ce = ce_loss(y_true, y_pred)
        return bayes_cross_entropy(y_true, ce, n_classes)

    return loss


model.compile(
    tf.keras.optimizers.Adam(lr=1e-6), #trying different optimizer!
    loss='categorical_crossentropy',
    metrics=['acc']
)

# --------- Training Model ---------
# Callbacks for training
def lr_schedule_fn(epoch, lr):
    return lr_schedule[epoch]

learning_rate_scheduler = LearningRateScheduler(
    lr_schedule_fn,
    verbose=1,
)

callbacks = [
#    learning_rate_scheduler,
    ModelCheckpoint(save_dir + model_name + ".h5",
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True),
    CSVLogger(save_dir + model_name + "_training.csv"),
]

if train_model:
    history = model.fit_generator(
        train_generator,
	epochs=epochs,
	verbose=1,
	callbacks=callbacks,
	validation_data=validation_generator,
	workers=1,
	use_multiprocessing=False
    )

    # store training progress for viz
    pickle.dump(history.history, open(save_dir + 'history_' + model_name + '.pkl', 'wb'))

    # load best weights
    model.load_weights(save_dir + model_name + ".h5")

# Calculate training roc_auc of best model
train_y_pred = model.predict( #_generator(
    validation_generator.next(),
    verbose=1
)
# roc_auc takes prob of positive class as input
train_y_pred = train_y_pred[:, 1]
train_y_true = validation_generator.classes[:32]
train_auc_score = roc_auc_score(train_y_true, train_y_pred)
print(train_y_pred)
print(train_y_true)
print("AUC score {:.5f}".format(train_auc_score))

# Calculate train accuracy and roc_auc
test_y_pred = model.predict_generator(
    test_generator,
    verbose=1
)
test_y_pred = test_y_pred[:, 1]
test_y_true = test_generator.classes
test_auc_score = roc_auc_score(test_y_true, test_y_pred)

print("AUC score {:.5f}".format(test_auc_score))
