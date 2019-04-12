"""
logistic regression with L2 regularization by batch gradient descent.
"""
import numpy as np
import tensorflow as tf
import time

from utils import read_data, write_to_file

TRAIN_SET_SIZE = 499
TEST_SET_SIZE = 343
FEATURE_SIZE = 310
NUM_CLASS = 3
MAX_EPOCH = 5000

LEARNING_RATE = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
HIDDEN_UNITS = [8, 16, 32, 64, 128, 256]
# LAMBDA = [0.01, 0.04, 0.08, 0.1, 0.5, 1]  # regularization coefficient.

log_path = './logs/'
data_path = './data/'


def build_model(lr, units):
    # Placeholders
    features = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    labels = tf.placeholder(tf.float32, [None, NUM_CLASS])

    with tf.variable_scope('eval_net_'):
        hl = tf.contrib.layers.fully_connected(features, units, activation_fn=tf.nn.relu)
        out_softmax = tf.contrib.layers.fully_connected(hl, NUM_CLASS, activation_fn=tf.nn.softmax)

    with tf.variable_scope('loss'):
        loss = -tf.reduce_sum(labels * tf.log(out_softmax))

    with tf.variable_scope('train_accuracy'):
        # train.
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)
        # accuracy evaluation.
        correct_prediction = tf.equal(tf.argmax(out_softmax, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return features, labels, out_softmax, train_op, loss, accuracy


def train(lr, units):
    # Read data.
    train_data, train_label, test_data, test_label = read_data(data_path)
    train_label = np.eye(3)[train_label.reshape(-1)]
    test_label = np.eye(3)[test_label.reshape(-1)]
    features, labels, out_softmax, train_op, loss, accuracy = build_model(lr, units)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Training start ...')
        for epoch_i in range(MAX_EPOCH):
            loss_epoch = 0
            for batch_i in range(10):
                batch_train_data, batch_train_label = train_data[batch_i*50:(batch_i+1)*50], \
                                                      train_label[batch_i*50:(batch_i+1)*50]
                _, loss_step = sess.run([train_op, loss],
                                        feed_dict={features: batch_train_data, labels: batch_train_label})
                loss_epoch += loss_step

            # if epoch_i % 50 == 0:
            #     print(epoch_i, ' | ', loss_epoch)

            if epoch_i % 200 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: test_data, labels: test_label})
                content = 'epoch, test accuracy = {0} | {1:.4%}'.format(epoch_i, train_accuracy)
                print(content)

        # Test trained model
        final_accuracy = sess.run(accuracy, feed_dict={features: test_data, labels: test_label})
        print('test accuracy = %10.4f' % final_accuracy)

        return final_accuracy


def main():
    classify_time = time.time()

    # Train and test the model.
    write_to_file(log_path + 'accuracy.txt', 'accuracy', True)
    accuracy = train(5e-6, 128)  # AdamOptimizer 1e-5 slow; 1e-4 normal; 1e-3 diverge
    # for lr in LEARNING_RATE:
    #     for unit in HIDDEN_UNITS:
    #         accuracy = train(lr, unit)
    #         write_to_file(log_path + 'accuracy.txt',
    #                       'lr:{0} | units: {1} | accuracy: {2}\n'.format(lr, unit, accuracy), False)
    classified_time = time.time() - classify_time
    print('Classified time: {0:.4f} seconds.'.format(classified_time))


if __name__ == '__main__':
    main()
