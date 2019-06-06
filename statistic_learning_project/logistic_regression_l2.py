"""
logistic regression with L2 regularization by batch gradient descent.
"""
import numpy as np
import tensorflow as tf
import time

from utils import read_test, read_train, write_to_file, write_to_csv

TRAIN_SET_SIZE = 7800
TEST_SET_SIZE = 15600
TRAIN_SIZE = 5200
VALID_SIZE = 1800
TEST_SIZE = 800
FEATURE_SIZE = 4096
NUM_CLASS = 12
MAX_EPISODES = 3000
LEARNING_RATE = 0.0001
LAMBDA = [0.01, 0.04, 0.08, 0.1, 0.5, 1]  # regularization coefficient.

log_path = './logs/lr_l2/'
output_path = './logs/lr_l2/submission.csv'


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


def build_model(regu_lambda, if_nm=True):
    # Placeholders
    features = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    labels = tf.placeholder(tf.float32, [None, NUM_CLASS])
    # Variables
    w = tf.Variable(tf.random_normal([FEATURE_SIZE, NUM_CLASS], mean=0.0, stddev=0.01))
    b = tf.Variable(tf.zeros([NUM_CLASS]))
    # softmax predictions
    y_pred = tf.matmul(features, w) + b
    y_pred_softmax = tf.nn.softmax(y_pred)  # for prediction
    # loss function and train step
    if if_nm is True:
        loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(y_pred_softmax, 1e-10, 1.0))) + \
               regu_lambda * tf.nn.l2_loss(w) + regu_lambda * tf.nn.l2_loss(b)
    else:
        loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(y_pred_softmax, 1e-10, 1.0)))
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
    # accuracy evaluation
    correct_prediction = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return features, labels, y_pred_softmax, train_step, loss, accuracy


def train(train_features, train_labels, valid_features, valid_labels, regu_lambda):
    features, labels, y_pred_softmax, train_step, loss, accuracy = build_model(regu_lambda)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Training start ...')
        local_log_path = log_path + str(regu_lambda) + '/lr_l2_log.txt'
        write_to_file(local_log_path, 'Training logs\n', True)
        batch_xs_train, batch_ys_train = train_features, np.eye(NUM_CLASS)[train_labels]
        batch_xs_valid, batch_ys_valid = valid_features, np.eye(NUM_CLASS)[valid_labels]
        for episode_i in range(2001):
            _, loss_step = sess.run([train_step, loss], feed_dict={features: batch_xs_train, labels: batch_ys_train})
            print(episode_i, ' | ', loss_step)
            write_to_file(local_log_path, str(episode_i) + ' | ' + str(loss_step) + '\n', False)

            if episode_i % 200 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: batch_xs_valid, labels: batch_ys_valid})
                str_i = 'step, loss, accuracy = {0} | {1:.4f} | {2:.4%}'.format(episode_i, loss_step, train_accuracy)
                print(str_i)
                write_to_file(local_log_path, str_i + '\n', False)

        # Test trained model
        final_accuracy = sess.run(accuracy, feed_dict={features: batch_xs_valid, labels: batch_ys_valid})
        print('final accuracy = %10.4f' % final_accuracy)

        return final_accuracy


def test(train_features, train_labels, test_features, optimal_lambda):
    features, labels, y_pred_softmax, train_step, loss, accuracy = build_model(optimal_lambda)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # print('Training start ...')
        local_log_path = log_path + 'lr_l2_log.txt'
        write_to_file(local_log_path, 'Training logs\n', True)
        train_time = time.time()

        batch_xs, batch_ys = train_features, np.eye(NUM_CLASS)[train_labels]
        for episode_i in range(MAX_EPISODES):
            _, loss_step = sess.run([train_step, loss], feed_dict={features: batch_xs, labels: batch_ys})
            print(episode_i, ' | ', loss_step)
            write_to_file(local_log_path, str(episode_i) + ' | ' + str(loss_step) + '\n', False)

            if episode_i % 200 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: batch_xs, labels: batch_ys})
                str_i = 'step, loss, accuracy = {0} | {1:.4f} | {2:.4%}'.format(episode_i, loss_step, train_accuracy)
                print(str_i)
                write_to_file(local_log_path, str_i + '\n', False)

        write_to_file(local_log_path, 'Training times {0}s\n'.format(time.time()-train_time), False)

        test_time = time.time()

        result_list = ['id,categories\n']
        y_pred = sess.run(y_pred_softmax, feed_dict={features: test_features})
        pred_cls = np.argmax(y_pred, 1)
        for i in range(len(pred_cls)):
            result_list.append(str(i) + ',' + str(pred_cls[i]) + '\n')

        write_to_file(local_log_path, 'Test times {0}s\n'.format(time.time() - test_time), False)

        write_to_csv(output_path, result_list)

        # # sess.run() one by one method (not recommended, it can slow sess.run()).
        # for i in range(TEST_SET_SIZE):  # 100:TEST_SET_SIZE
        #     test_image = test_features[i]
        #     test_image = np.expand_dims(test_image, axis=0)
        #     y_pred = sess.run(y_pred_softmax, feed_dict={features: test_image})
        #     pred_cls = np.argmax(y_pred, 1)
        #     print(i, ' | ', pred_cls[0])
        #     write_to_file(output_path, str(i) + ',' + str(pred_cls) + '\n', False)


def main():
    # read features and labels.
    test_features_all, train_features_all, train_labels_all = load_data()
    train_data = np.concatenate((train_features_all, train_labels_all[:, np.newaxis]), axis=1)
    np.random.shuffle(train_data)  # Reshuffle the whole training set.
    seg_1 = TRAIN_SIZE
    seg_2 = TRAIN_SIZE + VALID_SIZE
    train_features, train_labels = train_data[:seg_1, :-1], train_data[:seg_1, -1].astype(int)
    valid_features, valid_labels = train_data[seg_1:seg_2, :-1], train_data[seg_1:seg_2, -1].astype(int)
    test_features, test_labels = train_data[seg_2:, :-1], train_data[seg_2:, -1].astype(int)

    classify_time = time.time()

    # select the optimal lambda.
    accuracy = []
    for regu_lambda in LAMBDA:
        accuracy_term = train(train_features, train_labels, valid_features, valid_labels, regu_lambda)
        accuracy.append(accuracy_term)

    optimal_lambda = LAMBDA[int(np.argmax(accuracy))]
    accuracy_test = train(train_features, train_labels, test_features, test_labels, optimal_lambda)

    write_to_file('./logs/lr_l2/optimal_lambda.txt', str(accuracy) + '\n' + str(optimal_lambda) + '\n', True)
    write_to_file('./logs/lr_l2/accuracy_test.txt', str(accuracy_test), True)

    # optimal_lambda = 0.5
    # accuracy_test = train(train_features, train_labels, test_features, test_labels, optimal_lambda)
    # write_to_file('./logs/lr_l2/accuracy_test.txt', str(accuracy_test), True)
    test(train_features_all, train_labels_all, test_features_all, optimal_lambda)

    classified_time = time.time() - classify_time
    print('Classified time: {0:.4f} seconds.'.format(classified_time))


if __name__ == '__main__':
    main()
