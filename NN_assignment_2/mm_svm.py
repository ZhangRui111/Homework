import numpy as np
import time

from sklearn import svm, preprocessing
from sklearn.utils import shuffle
from utils import min_max_labels_random, min_max_labels_prior, write_to_file


# # data paths
PATH_TRAIN_DATA = './data/train_data.npy'
PATH_TRAIN_LABEL = './data/train_label.npy'
PATH_TEST_DATA = './data/test_data.npy'
PATH_TEST_LABEL = './data/test_label.npy'
NUM_CLS = 3
RANDOM_DECOM = True  # Whether task decomposition randomly or task decomposition with prior knowledge strategies.


def main():
    start = time.time()
    # # load data.
    train_data = np.load(PATH_TRAIN_DATA)
    train_label = np.load(PATH_TRAIN_LABEL)
    test_data = np.load(PATH_TEST_DATA)
    test_label = np.load(PATH_TEST_LABEL)

    # # shuffle the data
    train_data, train_label = shuffle(train_data, train_label)
    # test_data, test_label = shuffle(test_data, test_label)

    # # image preprocessing.
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    # # range label in [0,2]
    train_label += 1
    test_label += 1

    # # use a fraction of training data
    # train_data = train_data[0: 10000, :]
    # train_label = train_label[0: 10000]

    # # prepare label for min_max_svm
    if RANDOM_DECOM:
        train_datas, train_labels = min_max_labels_random(train_data, train_label)
        log_name = 'random'
    else:
        train_datas, train_labels = min_max_labels_prior(train_data, train_label)
        log_name = 'prior'
    # # ------------------------------------------------------- # #
    # # train_* is a three elements list:
    # #   / 1st element: two elements list
    # #   |     1st element: ndarray with shape (24843, 310) --
    # #   |         half of them are class 1 (denoted as P_1),
    # #   |         half of them are others, i.e., class 2 or class 3 (denoted as N_1).
    # #   |     2nd element: ndarray with shape (24844, 310) --
    # #  <          half of them are class 1 (denoted as P_2),
    # #   |         half of them are others, i.e., class 2 or class 3 (denoted as N_2).
    # #   |
    # #   |                  * P_1 == P_2 (all data of class 1);
    # #   |                  * N_1 + N_2 == all data of class 2 and class 3.
    # #   | 2nd element: ...
    # #   \ 3rd element: ...
    # # ------------------------------------------------------- # #
    print('Training...')
    svms = []
    for cls in range(NUM_CLS):
        print('training for class {}'.format(cls))
        sub_svms = []
        for i in range(2):
            clf = svm.SVC(kernel='rbf', C=2, probability=True)
            clf.fit(train_datas[cls][i], train_labels[cls][i])
            sub_svms.append(clf)
        svms.append(sub_svms)

    # # --------------- new version: svms.predict_proba --------------- # #
    print('Testing...')
    predicts = []
    for cls in range(NUM_CLS):
        sub_predicts = []
        for i in range(2):
            cls_predict = svms[cls][i].predict_proba(test_data)
            sub_predicts.append(cls_predict[:, 1])

        sub_predict = np.stack(sub_predicts, axis=1)
        sub_predict = np.average(sub_predict, axis=1)
        np.save('./logs/p2_{0}_{1}_prediction.txt'.format(log_name, cls), sub_predict)
        predicts.append(sub_predict)
    # # --------------- new version: svms.predict_proba --------------- # #

    # # --------------- old version: svms.predict --------------- # #
    # print('Testing...')
    # predicts = []
    # for cls in range(NUM_CLS):
    #     sub_predicts = []
    #     for i in range(2):
    #         cls_predict = svms[cls][i].predict(test_data)
    #         sub_predicts.append(cls_predict)
    #
    #     sub_predict = np.stack(sub_predicts, axis=1)
    #     sub_predict = np.min(sub_predict, axis=1)
    #     np.save('./logs/{}.npy'.format(cls), sub_predict)
    #     predicts.append(sub_predict)
    # # --------------- old version: svms.predict --------------- # #

    predict = np.stack(predicts, axis=1)
    predict = np.argmax(predict, axis=1)

    n_accuracy = np.where(np.equal(predict, test_label))[0]
    accuracy = n_accuracy.shape[0] / test_data.shape[0]
    print('final accuracy {}'.format(accuracy))

    cls_time = time.time() - start
    write_to_file('./logs/p2_{}_accuracy.txt'.format(log_name), '{}\n'.format(accuracy), False)
    write_to_file('./logs/p2_{}_time.txt'.format(log_name), '{}\n'.format(cls_time), True)


if __name__ == '__main__':
    main()
