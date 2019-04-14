'''
This will analyze the distribution of uncertainty.
'''

import sys
import numpy as np
import pickle

def load_stochastic_predictions(filename):
    with open(filename, 'rb') as f:
        probs_mc = pickle.load(f)['stoch_out']
        assert ((0.0 <= probs_mc) & (probs_mc <= 1.0 + 1e-6)).all()
        return probs_mc

def binary_probs(probs, min_positive_level):
    if probs.shape[1] == 2:
        return np.squeeze(probs[:, 1:])
    elif probs.shape[1] == 5:
        return probs[:, min_positive_level:].sum(axis=1)

def posterior_statistics(probs_mc_bin):
    predictive_mean = probs_mc_bin.mean(axis=1)
    predictive_std = probs_mc_bin.std(axis=1)
    assert (0.0 <= predictive_std).all()
    return predictive_mean, predictive_std

def main(pickle_file, min_positive_level):
    probs_mc = load_stochastic_predictions(pickle_file)
    probs_mc_bin = binary_probs(probs_mc, min_positive_level)
    pred_mean, pred_std = posterior_statistics(probs_mc_bin)
    uncertainties = list(pred_std)
    above = dict()
    for x in uncertainties:
        threshold_key = 5
        while x*100 >= threshold_key:
            above[threshold_key] = above.get(threshold_key, 0) + 1
            threshold_key += 5
    print('Average uncertainty: %.2f' % (sum(uncertainties) / len(uncertainties)))
    print()
    print('Distribution:')
    for k, v in above.items():
        print('%.2f%% are above %.2f' % (100*v/len(uncertainties), k/100.0))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print('  python analyze_uncertainty_dist.py <predictions.pkl> <min_positive_level>')
        exit()
    pickle_file = sys.argv[1]
    min_positive_level = int(sys.argv[2])
    main(pickle_file, min_positive_level)
