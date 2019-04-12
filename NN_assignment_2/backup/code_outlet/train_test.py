import timeit
import numpy as np
from sklearn import svm, preprocessing
from backup.code_outlet.preprocess import one_vs_rest_label, min_max_data_label

# data paths
PATH_TRAIN_DATA = '../../data/train_data.npy'
PATH_TRAIN_LABEL = '../../data/train_label.npy'
PATH_TEST_DATA = '../../data/test_data.npy'
PATH_TEST_LABEL = '../../data/test_label.npy'
# NUM CLASS
NUM_CLS = 3
# PARTIAL DATA
PARTIAL = True


def one_vs_rest_train_test():
    # read train data
    train_data = np.load(PATH_TRAIN_DATA)
    train_label = np.load(PATH_TRAIN_LABEL)
    test_data = np.load(PATH_TEST_DATA)
    test_label = np.load(PATH_TEST_LABEL)
    # preprocess
    # train_data = preprocessing.scale(train_data)
    # test_data = preprocessing.scale(test_data)
    # range label in [0,2]
    train_label += 1
    test_label += 1
    # use a fraction of training data
    if PARTIAL:
        data_inds = np.arange(0, train_data.shape[0], 1)
        np.random.shuffle(data_inds)
        data_inds = data_inds[:6000]
        train_data = train_data[data_inds]
        train_label = train_label[data_inds]
    # prepare label for one-vs-rest
    train_labels = one_vs_rest_label(train_label)

    print('start training...')
    # train
    svms = []
    for cls in range(NUM_CLS):
        start = timeit.timeit()
        print('training for class {}'.format(cls))
        clf = svm.SVC(C=2, kernel='rbf', probability=True)
        clf.fit(train_data, train_labels[cls])
        end = timeit.timeit()
        print('training for class {} finished'.format(cls))
        svms.append(clf)

    print('testing...')
    # test
    predicts = []
    for cls in range(NUM_CLS):
        cls_predict = svms[cls].predict_proba(test_data)
        predicts.append(cls_predict[:, 1])

    predicts = np.stack(predicts, axis=1)
    predict = np.argmax(predicts, axis=1)
    accuracy = np.where(np.equal(predict, test_label))[0]
    accuracy = accuracy.shape[0] / test_data.shape[0]
    print('final accuracy {}'.format(accuracy))


def min_max_train_test():
    # read train data
    train_data = np.load(PATH_TRAIN_DATA)
    train_label = np.load(PATH_TRAIN_LABEL)
    test_data = np.load(PATH_TEST_DATA)
    test_label = np.load(PATH_TEST_LABEL)
    # preprocess
    # train_data = preprocessing.scale(train_data)
    # test_data = preprocessing.scale(test_data)
    # range label in [0,2]
    train_label += 1
    test_label += 1
    # use a fraction of training data
    if PARTIAL:
        data_inds = np.arange(0, train_data.shape[0], 1)
        np.random.shuffle(data_inds)
        data_inds = data_inds[:6000]
        train_data = train_data[data_inds]
        train_label = train_label[data_inds]
    # prepare label for one-vs-rest
    train_datas, train_labels = min_max_data_label(train_data, train_label)

    print('start training...')
    # train
    svms = []
    for cls in range(NUM_CLS):
        start = timeit.timeit()
        print('training for class {}'.format(cls))
        sub_svms = []
        for i in range(2):
            print('{}...'.format(i))
            clf = svm.SVC(C=2, kernel='rbf', probability=True)
            clf.fit(train_datas[cls][i], train_labels[cls][i])
            end = timeit.timeit()
            sub_svms.append(clf)
        print('training for class {} finished'.format(cls))
        svms.append(sub_svms)

    print('testing...')
    # test
    predicts = []
    for cls in range(NUM_CLS):
        sub_predicts = []
        for i in range(2):
            cls_predict = svms[cls][i].predict_proba(test_data)
            sub_predicts.append(cls_predict[:, 1])

        sub_predict = np.stack(sub_predicts, axis=1)
        sub_predict = np.average(sub_predict, axis=1)
        # np.save('{}.npy'.format(2), sub_predict)
        predicts.append(sub_predict)

    predict = np.stack(predicts, axis=1)
    predict = np.argmax(predict, axis=1)

    accuracy = np.where(np.equal(predict, test_label))[0]
    accuracy = accuracy.shape[0] / test_data.shape[0]
    print('final accuracy {}'.format(accuracy))


if __name__ == '__main__':
    one_vs_rest_train_test()
