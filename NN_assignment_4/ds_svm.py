"""
Solve the domain shift problem in the given dataset SEED
using support vector machines (SVMs).
"""
import numpy as np
import time

from sklearn import svm, preprocessing
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.utils import shuffle
from utils import read_data


def domain_adaption_svm(s_data, s_labels, t_data, t_labels):
    source_data, source_labels, target_data, target_labels = s_data, s_labels, t_data, t_labels
    # # shuffle the data
    source_data, source_labels = shuffle(source_data, source_labels)
    # # data normalization.
    source_data = preprocessing.scale(source_data)
    target_data = preprocessing.scale(target_data)

    # # range label in [0,2]
    source_labels += 1
    target_labels += 1

    # # use a fraction of training data
    source_data = source_data[0: 10000, :]
    source_labels = source_labels[0: 10000]

    # # one-vs-one multi-class SVM
    clf = OneVsOneClassifier(svm.SVC(kernel='rbf', C=2)).fit(source_data, source_labels)
    # # one-vs-rest multi-class SVM
    # clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=2)).fit(source_data, source_labels)
    cls_predict = clf.predict(target_data)
    n_accuracy = np.where(np.equal(cls_predict, target_labels))[0]
    accuracy = n_accuracy.shape[0] / target_labels.shape[0]

    return accuracy


def main():
    all_data, all_labels = read_data('./data/EEG_X.mat', './data/EEG_Y.mat')
    assert len(all_data) == len(all_labels)
    size = len(all_data)
    source_inds = []
    target_inds = []
    all_accuracy = []

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

        accuracy = domain_adaption_svm(source_data, source_labels, target_data, target_labels)
        print('{0} -- final accuracy {1}'.format(i, accuracy))
        all_accuracy.append(accuracy)

        source_inds.clear()
        target_inds.clear()

    print(all_accuracy)
    print('Average accuracy over 15 subjects is {}'.format(sum(all_accuracy) / len(all_accuracy)))


if __name__ == '__main__':
    main()
