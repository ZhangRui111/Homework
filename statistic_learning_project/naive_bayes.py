import numpy as np
import time

from scipy.stats import multivariate_normal as mvn
from utils import read_test, read_train, write_to_file, write_to_csv

log_path = './logs/nb/nb_log.txt'
output_path = './logs/nb/submission.csv'

TRAIN_SET_SIZE = 7800
TEST_SET_SIZE = 15600
VALID_SIZE = 2800  # subdivide training set into training/valid set.
FEATURE_SIZE = 4096
NUM_CLASS = 12


def load_data(if_norm=True):
    print('Loading data set ...')
    load_time = time.time()
    test_data = read_test('./data/test.csv')
    test_data = np.array(test_data)
    train_data = read_train('./data/train.csv')
    train_data = np.array(train_data)
    loaded_time = time.time() - load_time
    print('test_data: {0}, train_data shape: {1}'.format(test_data.shape, train_data.shape))
    train_features, train_labels = train_data[:, 1:-1], train_data[:, -1].astype(int)
    test_features = test_data[:, 1:]
    if if_norm is True:
        test_min, test_max = test_features.min(), test_features.max()
        test_features_norm = (test_features - test_min) / (test_max - test_min)
        train_min, train_max = train_features.min(), train_features.max()
        train_features_norm = (train_features - train_min) / (train_max - train_min)
        print('Data set loaded successfully in {0:.4f} seconds.'.format(loaded_time))
        return test_features_norm, train_features_norm, train_labels
    else:
        print('Data set loaded successfully in {0:.4f} seconds.'.format(loaded_time))
        return test_features, train_features, train_labels


class NaiveBayes(object):
    def __init__(self):
        self.gaussians = dict()
        self.priors = dict()

    def fit(self, X, Y, smoothing=1e-2):
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]  # type(current_x) is ndarray.
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            # mvn.logpdf(): multivariate_normal's log probability distribution function.
            # The probability of x occurs given normal distribution's mean and Variance.
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


def main():
    # read features and labels.
    test_features_all, train_features_all, train_labels_all = load_data()
    train_data = np.concatenate((train_features_all, train_labels_all[:, np.newaxis]), axis=1)
    np.random.shuffle(train_data)  # Reshuffle the whole training set.
    length = TRAIN_SET_SIZE - VALID_SIZE
    train_features, train_labels = train_data[:length, :-1], train_data[:length, -1].astype(int)
    valid_features, valid_labels = train_data[length:, :-1], train_data[length:, -1].astype(int)

    print('Training start ...')

    # train phase (train & validation)
    model_valid = NaiveBayes()
    model_valid.fit(train_features, train_labels)

    accuracy_valid = model_valid.score(valid_features, valid_labels)
    print("Validation accuracy:", accuracy_valid)
    write_to_file(log_path, 'Validation accuracy: {0}s\n'.format(accuracy_valid), True)

    # classification phase.
    model = NaiveBayes()
    model.fit(train_features_all, train_labels_all)

    t0 = time.time()
    print("Train accuracy:", model.score(train_features_all, train_labels_all))
    print("Time to compute train accuracy: {0} second".format(time.time() - t0))
    write_to_file(log_path, 'Training time: {0}s\n'.format(time.time() - t0), False)

    t0 = time.time()
    result_list = ['id,categories\n']
    result = model.predict(test_features_all)
    for i in range(len(result)):
        result_list.append(str(i) + ',' + str(result[i]) + '\n')

    write_to_csv(output_path, result_list)

    print("Time to classify test data set: {0} second".format(time.time() - t0))
    write_to_file(log_path, 'Testing time: {0}s\n'.format(time.time() - t0), False)


if __name__ == '__main__':
    main()
