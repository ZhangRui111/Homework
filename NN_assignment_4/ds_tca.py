"""
Solve the domain shift problem in the given dataset SEED
using transfer component analysis (TCA).
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import time

from sklearn import svm, preprocessing
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.utils import shuffle
from utils import read_data, write_to_file


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """ Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """ Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        """
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e * e.T
        L = L / np.linalg.norm(L, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = my_kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, L, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """ Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = OneVsOneClassifier(svm.SVC(kernel='rbf', C=2)).fit(Xs_new, Ys)
        cls_predict = clf.predict(Xt_new)
        n_accuracy = np.where(np.equal(cls_predict, Yt))[0]
        accuracy = n_accuracy.shape[0] / Yt.shape[0]
        # acc = sklearn.metrics.accuracy_score(Yt, cls_predict)
        return accuracy, cls_predict


def my_kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def domain_adaption_tca(s_data, s_labels, t_data, t_labels, dim=30):
    source_data, source_labels, target_data, target_labels = s_data, s_labels, t_data, t_labels
    # # shuffle the data
    source_data, source_labels = shuffle(source_data, source_labels)
    target_data, target_labels = shuffle(target_data, target_labels)
    # # data normalization.
    source_data = preprocessing.scale(source_data)
    target_data = preprocessing.scale(target_data)

    # # range label in [0,2]
    source_labels += 1
    target_labels += 1

    # # use a fraction of training data
    source_data = source_data[0: 1000, :]
    source_labels = source_labels[0: 1000]
    target_data = target_data[0: 200, :]
    target_labels = target_labels[0: 200]
    
    # transfer component analysis
    tca = TCA(kernel_type='rbf', dim=dim, lamb=1, gamma=1)
    accuracy, prediction = tca.fit_predict(source_data, source_labels, target_data, target_labels)

    return accuracy


def main():
    all_data, all_labels = read_data('./data/EEG_X.mat', './data/EEG_Y.mat')
    assert len(all_data) == len(all_labels)
    size = len(all_data)
    source_inds = []
    target_inds = []
    all_accuracy = []
    dims = [5, 10, 20, 30, 50, 100]

    write_to_file('./logs/accuracy.txt', 'accuracy\n', True)
    for m in dims:
        for i in range(size):  # i -- the i_th subject is the target domain.
            for j in range(size):  # all subjects except the i_th make the source domain.
                if i != j:
                    source_inds.append(j)
                else:
                    target_inds.append(j)
            size_s = len(source_inds)
            for k in range(size_s):
                ind = source_inds.pop()
                if k == 0:
                    source_data = all_data[ind]
                    source_labels = all_labels[ind]
                else:
                    source_data = np.concatenate((source_data, all_data[ind]), axis=0)
                    source_labels = np.concatenate((source_labels, all_labels[ind]), axis=0)
            ind_t = target_inds.pop()
            target_data = all_data[ind_t]
            target_labels = all_labels[ind_t]
            source_labels = source_labels.ravel()
            target_labels = target_labels.ravel()

            i_accuracy = []
            for n in range(10):
                i_accuracy.append(domain_adaption_tca(source_data, source_labels, target_data, target_labels, dim=m))
            acc = sum(i_accuracy) / len(i_accuracy)
            print('{0} -- accuracy {1}'.format(i, acc))
            all_accuracy.append(acc)

            source_inds.clear()
            target_inds.clear()

        print(all_accuracy)
        write_to_file('./logs/accuracy.txt', '{0}: {1} -- {2}\n'.format(m, all_accuracy, sum(all_accuracy) / len(all_accuracy)), False)
        print('{0} -- Average accuracy over 15 subjects is {1}'.format(m, sum(all_accuracy) / len(all_accuracy)))
        all_accuracy.clear()


if __name__ == '__main__':
    main()
