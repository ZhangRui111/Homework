import numpy as np
import time

from sklearn import svm, preprocessing
from sklearn.utils import shuffle
from utils import one_vs_rest_labels, write_to_file


# # data paths
PATH_TRAIN_DATA = './data/train_data.npy'
PATH_TRAIN_LABEL = './data/train_label.npy'
PATH_TEST_DATA = './data/test_data.npy'
PATH_TEST_LABEL = './data/test_label.npy'
NUM_CLS = 3


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
    train_data = train_data[0: 10000, :]
    train_label = train_label[0: 10000]

    # # prepare label for one-vs-rest
    train_labels = one_vs_rest_labels(train_label)

    print('Training...')
    svms = []
    for cls in range(NUM_CLS):
        print('Training for class {}'.format(cls))
        clf = svm.SVC(kernel='rbf', C=2, probability=True)
        clf.fit(train_data, train_labels[cls])
        svms.append(clf)

    # # --------------- new version: svms.predict_proba --------------- # #
    print('Testing...')
    predicts = []
    for cls in range(NUM_CLS):
        cls_predict = svms[cls].predict_proba(test_data)
        predicts.append(cls_predict[:, 1])
    # # --------------- new version: svms.predict_proba --------------- # #

    # # --------------- old version: svms.predict --------------- # #
    # print('Testing...')
    # predicts = []
    # for cls in range(NUM_CLS):
    #     cls_predict = svms[cls].predict(test_data)
    #     predicts.append(cls_predict)
    # # --------------- old version: svms.predict --------------- # #

    predicts = np.stack(predicts, axis=1)
    predict = np.argmax(predicts, axis=1)
    n_accuracy = np.where(np.equal(predict, test_label))[0]
    accuracy = n_accuracy.shape[0] / test_data.shape[0]
    print('final accuracy {}'.format(accuracy))

    cls_time = time.time() - start
    write_to_file('./logs/p1_accuracy.txt', '{}\n'.format(accuracy), False)
    write_to_file('./logs/p1_time.txt', '{}\n'.format(cls_time), True)


if __name__ == '__main__':
    main()
