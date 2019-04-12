"""
knn algorithm with K's selection.
"""
import numpy as np
import time

from utils import read_test, read_train, write_to_file, write_to_csv

TRAIN_SET_SIZE = 7800
TEST_SET_SIZE = 15600
VALID_SIZE = 2800  # subdivide training set into training/validation set.
FEATURE_SIZE = 4096
NUM_CLASS = 12
K = [1, 3, 5, 7, 9, 11, 13, 15, 31, 63, 127]

log_path = './logs/knn/knn_log.txt'
out_put_path = './logs/knn/submission.csv'


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


def main():
    # read features and labels.
    test_features_all, train_features_all, train_labels_all = load_data()
    train_data = np.concatenate((train_features_all, train_labels_all[:, np.newaxis]), axis=1)
    np.random.shuffle(train_data)  # Reshuffle the whole training set.
    length = TRAIN_SET_SIZE - VALID_SIZE
    train_features, train_labels = train_data[:length, :-1], train_data[:length, -1].astype(int)
    valid_features, valid_labels = train_data[length:, :-1], train_data[length:, -1].astype(int)

    classify_time = time.time()

    # Find the optimal k.
    accuracy = np.zeros(len(K))
    for i in range(VALID_SIZE):  # 200:VALID_SIZE
        valid_image = valid_features[i]
        valid_image = np.expand_dims(valid_image, axis=0)
        pred = valid_image - train_features
        pred = np.sum(pred ** 2, axis=-1)
        pred_inds = np.argsort(pred)  # Returns the indices that would sort an array from small to big.
        ind = 0
        for k in K:
            pred_cls_array = train_labels[pred_inds[0:k]]
            pred_cls = np.argmax(np.bincount(pred_cls_array))
            if pred_cls == valid_labels[i]:
                accuracy[ind] += 1
            ind += 1
        print(i, ' | ', accuracy)

    write_to_file(log_path, 'accuracy list: {0}\n'.format(accuracy), True)
    optimal_k = K[int(np.argmax(accuracy))]
    write_to_file(log_path, 'optimal_k: {0}\n'.format(optimal_k), False)

    # Classify all test_data.
    test_time = time.time()
    result_list = ['id,categories\n']
    size = test_features_all.shape[0]
    for j in range(size):  # 100:TEST_SET_SIZE
        test_image = test_features_all[j]
        test_image = np.expand_dims(test_image, axis=0)
        pred = test_image - train_features_all
        pred = np.sum(pred ** 2, axis=-1)
        pred_inds = np.argsort(pred)  # Returns the indices that would sort an array from small to big.
        pred_cls_array = train_labels_all[pred_inds[0:optimal_k]]
        pred_cls = np.argmax(np.bincount(pred_cls_array))
        print(j, ' | ', pred_cls)
        result_list.append(str(j) + ',' + str(pred_cls) + '\n')
    write_to_file(log_path, 'test time: {0}s\n'.format(time.time()-test_time), False)

    write_to_csv(out_put_path, result_list)

    classified_time = time.time() - classify_time
    print('Classified time: {0:.4f} seconds.'.format(classified_time))


if __name__ == '__main__':
    main()
