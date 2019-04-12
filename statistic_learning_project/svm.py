"""
svm algorithm.
"""
import numpy as np
import time

from sklearn.svm import SVC
from utils import read_test, read_train, write_to_file, write_to_csv

TRAIN_SET_SIZE = 7800
TEST_SET_SIZE = 15600
TRAIN_SIZE = 5200
VALID_SIZE = 100
TEST_SIZE = 2500
K_FOLD_CV = 10
FEATURE_SIZE = 4096
NUM_CLASS = 12
BATCH_SIZE = 100

log_path = './logs/svm/svm_log.txt'
out_put_path = './logs/svm/submission.csv'


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


def reshuffle_data(train_data):
    np.random.shuffle(train_data)  # Reshuffle the whole training set.
    seg_1 = TRAIN_SIZE
    seg_2 = TRAIN_SIZE + VALID_SIZE
    train_features, train_labels = train_data[:seg_1, :-1], train_data[:seg_1, -1].astype(int)
    valid_features, valid_labels = train_data[seg_1:seg_2, :-1], train_data[seg_1:seg_2, -1].astype(int)
    test_features, test_labels = train_data[seg_2:, :-1], train_data[seg_2:, -1].astype(int)
    return train_features, train_labels, test_features, test_labels


def main():
    # read features and labels.
    test_features_all, train_features_all, train_labels_all = load_data()
    train_data = np.concatenate((train_features_all, train_labels_all[:, np.newaxis]), axis=1)
    np.random.shuffle(train_data)  # Reshuffle the whole training set.
    # seg_1 = TRAIN_SIZE
    # seg_2 = TRAIN_SIZE + VALID_SIZE
    # train_features, train_labels = train_data[:seg_1, :-1], train_data[:seg_1, -1].astype(int)
    # valid_features, valid_labels = train_data[seg_1:seg_2, :-1], train_data[seg_1:seg_2, -1].astype(int)
    # test_features, test_labels = train_data[seg_2:, :-1], train_data[seg_2:, -1].astype(int)

    K_train_data_list = []
    seg_length = int(TRAIN_SET_SIZE / K_FOLD_CV)
    for i in range(K_FOLD_CV):
        start = i * seg_length
        end = (i+1) * seg_length
        K_train_data_list.append(train_data[start:end, :])

    classify_time = time.time()

    # cross validation for hyper parameters.
    write_to_file(log_path, 'CV accuracy:\n', True)
    # all hyper parameters for cross validation.
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    penalties = [0.1, 0.5, 1, 2]
    cv_accuracy = []
    for kernal_type in kernels:
        for penalties in penalties:
            K_accuracy = []  # use to hold temporary train accuracy for mean.
            for i in range(K_FOLD_CV):
                K_train_data = None
                # get train data and test data separately.
                for j in range(K_FOLD_CV):
                    if j != i:
                        if K_train_data is None:
                            K_train_data = K_train_data_list[j]
                        else:
                            K_train_data = np.concatenate((K_train_data, K_train_data_list[j]))
                K_test_data = K_train_data_list[i]
                train_features, train_labels = K_train_data[:, :-1], K_train_data[:, -1].astype(int)
                test_features, test_labels = K_test_data[:, :-1], K_test_data[:, -1].astype(int)

                # print(train_features.shape)
                # train the model.
                svm_model_linear = SVC(kernel=kernal_type, C=float(penalties)).fit(train_features, train_labels)
                # get the train accuracy.
                accuracy_test = svm_model_linear.score(test_features, test_labels)
                print('kernel: {0} | penalty: {1} | accuracy: {2}'.format(kernal_type, penalties, accuracy_test))
                K_accuracy.append(accuracy_test)

            cv_accuracy.append(sum(K_accuracy)/len(K_accuracy))

        write_to_file(log_path, '\nCV accuracy: {0}\n'.format(cv_accuracy), True)

    write_to_file(log_path, '\nCV accuracy: {0}\n'.format(cv_accuracy), True)

    # train_features, train_labels, test_features, test_labels = reshuffle_data(train_data)
    # p_kernel, p_C, p_gamma, p_degree = 'poly', 1, 'auto', 3
    # svm_model_linear = SVC(kernel=p_kernel, C=p_C, gamma=p_gamma, degree=p_degree).fit(train_features, train_labels)
    # accuracy_test = svm_model_linear.score(test_features, test_labels)
    # print('accuracy: {0} | C: {1} | gamma: {2} | degree: {3}'.format(accuracy_test, p_C, p_gamma, p_degree))
    #
    # train_features, train_labels, test_features, test_labels = reshuffle_data(train_data)
    # p_kernel, p_C, p_gamma, p_degree = 'poly', 10, 'auto', 3
    # svm_model_linear = SVC(kernel=p_kernel, C=p_C, gamma=p_gamma, degree=p_degree).fit(train_features, train_labels)
    # accuracy_test = svm_model_linear.score(test_features, test_labels)
    # print('accuracy: {0} | C: {1} | gamma: {2} | degree: {3}'.format(accuracy_test, p_C, p_gamma, p_degree))
    #
    # train_features, train_labels, test_features, test_labels = reshuffle_data(train_data)
    # p_kernel, p_C, p_gamma, p_degree = 'poly', 50, 'auto', 3
    # svm_model_linear = SVC(kernel=p_kernel, C=p_C, gamma=p_gamma, degree=p_degree).fit(train_features, train_labels)
    # accuracy_test = svm_model_linear.score(test_features, test_labels)
    # print('accuracy: {0} | C: {1} | gamma: {2} | degree: {3}'.format(accuracy_test, p_C, p_gamma, p_degree))

    # predict for the test set.
    ind = cv_accuracy.index(max(cv_accuracy))
    kernel_selected = kernels[ind // 4]
    penalty_selected = penalties[ind % 4]
    svm_model_SVC = SVC(kernel=kernel_selected, C=penalty_selected, gamma='auto')
    svm_model = svm_model_SVC.fit(train_features_all, train_labels_all)
    svm_predictions = svm_model.predict(test_features_all)

    # write result to csv file.
    result_list = ['id,categories\n']
    for j in range(len(svm_predictions)):  # 100:TEST_SET_SIZE
        result_list.append(str(j) + ',' + str(svm_predictions[j]) + '\n')
    write_to_csv(out_put_path, result_list)

    classified_time = time.time() - classify_time
    print('Classified time: {0:.4f} seconds.'.format(classified_time))


if __name__ == '__main__':
    main()
