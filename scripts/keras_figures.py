from collections import OrderedDict
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import statsmodels.nonparametric.api as smnp

from util import roc_curve_plot
from util import bootstrap
from util import balance_classes

from matplotlib import rcParams

predictions_dir = "../predict_output/"
train_dir = "../../output/"
test_dir = "../../output_test/"
save_dir = "figures/"

messidor_dir = '../../messidor/'

DATA = {
    'KaggleDR_train':
        {'LABELS_FILE': train_dir + 'trainLabels.csv',
         'IMAGE_PATH': train_dir,
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild DR'),
                               (2, 'moderate DR'),
                               (3, 'severe DR'),
                               (4, 'proliferative DR')]),
         'min_percentile': 50,
         'n_bootstrap': 10000},
    'KaggleDR':
        {'LABELS_FILE': test_dir + 'testLabels.csv',
         'IMAGE_PATH': test_dir,
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild DR'),
                               (2, 'moderate DR'),
                               (3, 'severe DR'),
                               (4, 'proliferative DR')]),
         'min_percentile': 50,
         'n_bootstrap': 10000},
    'Messidor':
        {'LABELS_FILE': messidor_dir + 'messidor.csv',
         'IMAGE_PATH': messidor_dir,
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild non-prolif'),
                               (2, 'severe non-prolif'),
                               (3, 'most serious')]),
         'min_percentile': 50,
         'n_bootstrap': 1000}
}

CONFIG = {
    'BCNN_mildDR_Kaggle_train': dict(
        [('net', 'BCNN'),
         ('dataset', 'Kaggle train'),
         ('predictions', predictions_dir +
          'full_mc_100_kaggledr_0vs1234_bcnn.pkl'),
         ('disease_onset', 1)] + list(
        DATA['KaggleDR_train'].items())),

    'BCNN_mildDR_Kaggle': dict(
        [('net', 'BCNN'),
         ('dataset', 'Kaggle'),
         ('predictions', predictions_dir + 'full_mc_100_kaggledr_0vs1234_bcnn.pkl'),
         ('disease_onset', 1)] + list(
        DATA['KaggleDR'].items())),

    'BCNN_moderateDR_Kaggle': dict(
        [('net', 'BCNN'),
         ('dataset', 'Kaggle'),
         ('predictions', predictions_dir + 'full_mc_100_kaggledr_01vs234_bcnn.pkl'),
         ('disease_onset', 2)] + list(
        DATA['KaggleDR'].items())),

    'BCNN_moderateDR_Messidor': dict(
        [('net', 'BCNN'),
         ('dataset', 'Messidor'),
         ('predictions', predictions_dir +
          'mc_100_messidor_bcnn.pkl'),
         ('disease_onset', 2)] + list(
        DATA['Messidor'].items())),

    'JFnet_mildDR_Kaggle': dict(
        [('net', 'JFnet'),
         ('dataset', 'Kaggle'),
         ('predictions', predictions_dir +
          'c9ade47_100_mc_KaggleDR_test_JFnet.pkl'), #TODO
         ('disease_onset', 1)] + list(
        DATA['KaggleDR'].items())),

    'JFnet_moderateDR_Kaggle': dict(
        [('net', 'JFnet'),
         ('dataset', 'Kaggle'),
         ('predictions', predictions_dir +
          'c9ade47_100_mc_KaggleDR_test_JFnet.pkl'), #TODO
         ('disease_onset', 2)] + list(
        DATA['KaggleDR'].items())),
}

# ---- FIGURE PARAMETERS -----
rcParams.update({'figure.autolayout': True})

plt.ion()
sns.set_context('paper', font_scale=2)
sns.set_style('whitegrid')

FIGURE_WIDTH = 8.27 * 1.5 # 8.27 inch corresponds to A4

TAG = {0: 'healthy', 1: 'diseased'}
ONSET_TAG = {1: 'mild DR', 2: 'moderate DR'}

# ----- UTIL FUNCTIONS ------

# Set nrows to None later!
# This can limit the amount of data to make plotting faster
def load_labels(labels_file, nrows=9600):
    df_test = pd.read_csv(labels_file, nrows=nrows)
    y_test = df_test.level.values
    return y_test


def load_filenames(labels_file):
    df_test = pd.read_csv(labels_file)
    return df_test.image.values


def load_predictions(filename):
    """Load test predictions obtained with scripts/predict.py"""
    with open(filename, 'rb') as h:
        pred_test = pickle.load(h)
    probs = pred_test['det_out']
    probs_mc = pred_test['stoch_out']
    assert ((0.0 <= probs) & (probs <= 1.0 + 1e-6)).all()
    assert ((0.0 <= probs_mc) & (probs_mc <= 1.0 + 1e-6)).all()
    return probs, probs_mc


def binary_labels(labels, min_positive_level=1):
    labels_bin = np.zeros_like(labels)
    labels_bin[labels < min_positive_level] = 0
    labels_bin[labels >= min_positive_level] = 1
    return labels_bin


def binary_probs(probs, min_positive_level=1):
    n_classes = probs.shape[1]
    if n_classes == 5:
        return probs[:, min_positive_level:].sum(axis=1)
    elif n_classes == 2:
        return np.squeeze(probs[:, 1:])
    else:
        print('Unknown number of classes: %d. Aborting.' % n_classes)


def binary_entropy(p):
    assert p.ndim == 1
    return -(p * np.log2(p + 1e-6) + (1 - p) * np.log2((1 - p) + 1e-6))


def detection_task(y, probs, probs_mc, disease_level):
    y_diseased = binary_labels(y, disease_level)
    probs_diseased = binary_probs(probs, disease_level)
    probs_mc_diseased = binary_probs(probs_mc, disease_level)
    return y_diseased, probs_diseased, probs_mc_diseased


def mode(data):
    """Compute a kernel density estimate and return the mode"""
    if len(np.unique(data)) == 1:
        return data[0]
    else:
        kde = smnp.KDEUnivariate(data.astype('double'))
        kde.fit(cut=0)
        grid, y = kde.support, kde.density
        return grid[y == y.max()][0]


def posterior_statistics(probs_mc_bin):
    predictive_mean = probs_mc_bin.mean(axis=1)
    predictive_std = probs_mc_bin.std(axis=1)
    assert (0.0 <= predictive_std).all()
    return predictive_mean, predictive_std


def argmax_labels(probs):
    return (probs >= 0.5).astype(int)


def accuracy(y_true, probs):
    y_pred = argmax_labels(probs)
    assert len(y_true) == len(y_pred)
    return (y_true == y_pred).sum() / float(len(y_true))


def rel_freq(y, k):
    return (y == k).sum() / float(len(y))


def contralateral_agreement(y, config):
    """Get boolean array of contralateral label agreement

    Notes
    =====

    A very similar function is already there in datasets.py but here we want
    to work on indices and more importantly check for contralateral label
    agreement for a potentially binary label vector y for the corresponding
    disease detection problem.

    """

    if 'kaggle_dr' not in config['LABELS_FILE']:
        raise TypeError('Laterality not defined for %s'
                        % config['LABELS_FILE'])

    df = pd.read_csv(config['LABELS_FILE'])
    left = df.image.str.contains(r'\d+_left').values
    right = df.image.str.contains(r'\d+_right').values

    accepted_patients = (y[left] == y[right])
    accepted_images_left = df[left].image[accepted_patients]
    accepted_images_right = df[right].image[accepted_patients]
    accepted_images = pd.concat((accepted_images_left,
                                 accepted_images_right))
    return df.image.isin(accepted_images).values


def performance_over_uncertainty_tol(uncertainty, y, probs, measure,
                                     min_percentile, n_bootstrap):

    uncertainty_tol, frac_retain, accept_idx = \
        sample_rejection(uncertainty, min_percentile)

    p = np.zeros((len(uncertainty_tol),), dtype=[('value', 'float64'),
                                                 ('low', 'float64'),
                                                 ('high', 'float64')])
    p_rand = np.zeros((len(uncertainty_tol),), dtype=[('value', 'float64'),
                                                      ('low', 'float64'),
                                                      ('high', 'float64')])

    for i, ut in enumerate(uncertainty_tol):
        accept = accept_idx[i]
        rand_sel = np.random.permutation(accept)

        low, high = bootstrap([y[accept], probs[accept]], measure,
                              n_resamples=n_bootstrap, alpha=0.05)

        p['value'][i] = measure(y[accept], probs[accept])
        p['low'][i] = low.value
        p['high'][i] = high.value

        low, high = bootstrap([y[rand_sel], probs[rand_sel]], measure,
                              n_resamples=100, alpha=0.05)

        p_rand['value'][i] = measure(y[rand_sel], probs[rand_sel])
        p_rand['low'][i] = low.value
        p_rand['high'][i] = high.value

    return uncertainty_tol, frac_retain, p, p_rand


def sample_rejection(uncertainty, min_percentile,
                     maximum=None):
    if maximum is None:
        maximum = uncertainty.max()
    uncertainty_tol = np.linspace(np.percentile(uncertainty, min_percentile),
                                  maximum, 100)
    frac_retain = np.zeros_like(uncertainty_tol)
    n_samples = len(uncertainty)
    accept_indices = []
    for i, ut in enumerate(uncertainty_tol):
        accept = (uncertainty <= ut)
        accept_indices.append(accept)
        frac_retain[i] = accept.sum() / float(n_samples)

    return uncertainty_tol, frac_retain, accept_indices


# ---------- Fig 2 ----------
def prediction_vs_uncertainty(y, uncertainty, prediction,
                              title='', n_levels=250, balance=False,
                              ax121=None, ax122=None):
    ylabel = list(uncertainty.keys())[0]
    uncertainty = list(uncertainty.values())[0]
    xlabel = list(prediction.keys())[0]
    prediction = list(prediction.values())[0]

    print(type(prediction), prediction)
    print(type(xlabel), xlabel)
    print(type(ylabel), ylabel)
    print(type(uncertainty), uncertainty)

    if balance:
        y, (uncertainty, prediction) = balance_classes(y, [uncertainty,
                                                           prediction])

    error = prediction >= 0.5
    error = np.not_equal(y, prediction)

    plt.suptitle(title)

    if ax121 is None:
        ax121 = plt.subplot(1, 2, 1)

    ax121.set_title('(a) correct')
    sns.kdeplot(prediction[~error], uncertainty[~error],
                n_levels=n_levels, ax=ax121)
    ax121.set_ylabel(ylabel)
    ax121.set_xlabel(xlabel)
    ax121.set_xlim(0, 1.0)
    ax121.set_ylim(0, 0.25)

    if ax122 is None:
        ax122 = plt.subplot(1, 2, 2)

    ax122.set_title('(b) error')
    sns.kdeplot(prediction[error], uncertainty[error],
                n_levels=n_levels, ax=ax122)
    ax122.set_ylabel(ylabel)
    ax122.set_xlabel(xlabel)
    ax122.set_xlim(0, 1.0)
    ax122.set_ylim(0, 0.25)

    sns.despine(offset=10, trim=True)


def bayes_vs_softmax(save=True):
    config = CONFIG['BCNN_moderateDR_Kaggle']
    y = load_labels(config['LABELS_FILE'])
    probs, probs_mc = load_predictions(config['predictions'])
    y_bin, probs_bin, probs_mc_bin = detection_task(y, probs, probs_mc,
                                                    config['disease_onset'])
    _, pred_std = posterior_statistics(probs_mc_bin)
    uncertainty = {'$\sigma_{pred}$': pred_std}
    prediction = {'p(diseased | image)': probs_bin}

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    prediction_vs_uncertainty(y_bin, uncertainty, prediction,
                              title='', n_levels=250)
    name = 'sigma_vs_soft_' + config['net'] + '_' + \
        str(config['disease_onset']) + '_' + config['dataset']

    if save:
        fig.savefig(save_dir + name + ".png", bbox_inches='tight')

    return {name: fig}


# ---------- Fig 3 ----------
def acc_rejection_figure(y, y_score, uncertainties, config,
                         save=False, format='.png', fig=None):
    if fig is None:
        fig = plt.figure(figsize=(FIGURE_WIDTH,
                                  FIGURE_WIDTH / 2.0))

    colors = sns.color_palette()

    ax121 = plt.subplot(1, 2, 1)
    ax122 = plt.subplot(1, 2, 2)
    ax121.set_title('(a)')
    ax122.set_title('(b)')

    min_acc = 1.0
    for i, (k, v) in enumerate(uncertainties.items()):
        v_tol, frac_retain, acc, acc_rand = \
            performance_over_uncertainty_tol(v, y, y_score, accuracy, 0.0,
                                             config['n_bootstrap'])
        ax121.plot(v_tol, acc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax122.plot(frac_retain, acc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax121.fill_between(v_tol, acc['value'], acc['low'],
                           color=colors[i], alpha=0.3)
        ax121.fill_between(v_tol, acc['high'], acc['value'],
                           color=colors[i], alpha=0.3)
        ax122.fill_between(frac_retain, acc['value'], acc['low'],
                           color=colors[i], alpha=0.3)
        ax122.fill_between(frac_retain, acc['high'], acc['value'],
                           color=colors[i], alpha=0.3)
        if min_acc > min(min(acc['low']), min(acc_rand['low'])):
            min_acc = min(min(acc['low']), min(acc_rand['low']))

    ax121.set_ylim(min_acc, 1)
    ax122.set_ylim(min_acc, 1)
    ax122.set_xlim(0.1, 1.0)
    ax121.set_xlabel('tolerated model uncertainty')
    ax121.set_ylabel('accuracy')
    ax121.legend(loc='best')

    ax122.plot(frac_retain, acc_rand['value'], label='random referral',
               color=colors[i+1], linewidth=2)
    ax122.fill_between(frac_retain, acc_rand['value'], acc_rand['low'],
                       color=colors[i+1], alpha=0.3)
    ax122.fill_between(frac_retain, acc_rand['high'], acc_rand['value'],
                       color=colors[i+1], alpha=0.3)
    ax122.set_xlabel('fraction of retained data')
    ax122.legend(loc='best')

    sns.despine(offset=10, trim=True)

    name = 'acc_' + config['net'] + '_' + str(config['disease_onset']) + \
           '_' + config['dataset']

    if save:
        fig.savefig(save_dir + name + format, bbox_inches='tight')

    return {name: fig}

# ---------- Fig 6 ----------
def level_subplot(y_level, uncertainty, config,
                  ax=None):
    tol, frac_retain, accept_idx = sample_rejection(uncertainty, 0)
    LEVEL = config['LEVEL']
    p = {level: np.array([rel_freq(y_level[~accept], level)
                          for accept in accept_idx])
         for level in LEVEL}
    cum = np.zeros_like(tol)

    with sns.axes_style('white'):
        ax.set_title('Disease onset: %s'
                     % ONSET_TAG[config['disease_onset']])

        colors = {level: sns.color_palette("Blues")[level] for level in LEVEL}
        colors[0] = 'white'

        for level in LEVEL:
            ax.fill_between(tol, p[level] + cum, cum,
                            color=colors[level],
                            label='%d: %s' % (level, LEVEL[level]))
            if (level + 1) == config['disease_onset']:
                ax.plot(tol, p[level] + cum,
                        color='k', label='healthy/diseased boundary')
            cum += p[level]

        ax.set_xlim(min(tol), max(tol))
        ax.set_ylim(0, 1)

        ax.set_xlabel('tolerated model uncertainty')
        ax.set_ylabel('relative proportions within referred dataset')
        ax.legend(loc='lower left', prop={'size' : 12})


def level_figure(save=True):
    keys = ['BCNN_mildDR_Kaggle',
            'BCNN_moderateDR_Kaggle']
    title_prefix = ['(a)', '(b)']
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    for i, k in enumerate(keys):
        config = CONFIG[k]
        y = load_labels(config['LABELS_FILE'])
        probs, probs_mc = load_predictions(config['predictions'])
        _, _, probs_mc_bin = detection_task(y, probs, probs_mc,
                                            config['disease_onset'])
        _, pred_std = posterior_statistics(probs_mc_bin)

        ax = fig.add_subplot(1, 2, i + 1)
        level_subplot(y, pred_std, config, ax=ax)
        ax.set_title(title_prefix[i] + ' ' + ax.get_title())
        if i == 1:
            ax.set_ylabel('')

    if save:
        fig.savefig(save_dir + "level_figure.png", bbox_inches='tight')

    return {'level': fig}

# ---------- Main ----------
def main():
    config = CONFIG['BCNN_moderateDR_Kaggle']

    y = load_labels(config['LABELS_FILE'])
    images = load_filenames(config['LABELS_FILE'])
    probs, probs_mc = load_predictions(config['predictions'])
    y_bin, probs_bin, probs_mc_bin = detection_task(
        y, probs, probs_mc, config['disease_onset'])
    pred_mean, pred_std = posterior_statistics(probs_mc_bin)
    uncertainties = {'$\sigma_{pred}$': pred_std}

    bayes_vs_softmax()
    print("Figure 2 done")

    acc_rejection_figure(
        y_bin, pred_mean, uncertainties, config,
        save=True, format='.png'
    )
    print("Figure 3 done")

    level_figure()
    print("Figure 6 done")


if __name__ == '__main__':
    figures = main()
