"""
logistic regression with L2 regularization by batch gradient descent.
"""
import numpy as np
import os
import tensorflow as tf
import time

from utils import read_data, write_to_file, save_parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAIN_SET_SIZE = 499
TEST_SET_SIZE = 343
FEATURE_SIZE = 310
NUM_CLASS = 3
MAX_EPOCH = 5000

LEARNING_RATE = [5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3]
HIDDEN_UNITS = [8, 32, 128, 512, 2048]
# LAMBDA = [0.01, 0.04, 0.08, 0.1, 0.5, 1]  # regularization coefficient.

log_path = './logs/'
data_path = './data/'


def build_model(lr, units):
    # Placeholders
    features = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    labels = tf.placeholder(tf.float32, [None, NUM_CLASS])

    with tf.variable_scope('eval_net_{0}{1}'.format(lr, units)):
        hl = tf.contrib.layers.fully_connected(features, units, activation_fn=tf.nn.relu)
        out_softmax = tf.contrib.layers.fully_connected(hl, NUM_CLASS, activation_fn=tf.nn.softmax)

    with tf.variable_scope('loss{0}{1}'.format(lr, units)):
        loss = -tf.reduce_sum(labels * tf.log(out_softmax))

    with tf.variable_scope('train_accuracy{0}{1}'.format(lr, units)):
        # train.
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
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
    train_acc = []
    test_acc = []

    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver(max_to_keep=8)
    sess.run(tf.global_variables_initializer())

    print('Training start ...')
    print('---------------------------------------')
    for epoch_i in range(MAX_EPOCH):
        loss_epoch = 0

        for batch_i in range(10):
            batch_train_data, batch_train_label = train_data[batch_i*50:(batch_i+1)*50], \
                                                  train_label[batch_i*50:(batch_i+1)*50]
            _, loss_step = sess.run([train_op, loss],
                                    feed_dict={features: batch_train_data, labels: batch_train_label})
            loss_epoch += loss_step

        if epoch_i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={features: train_data, labels: train_label})
            content = 'epoch: {0} | training accuracy: {1:.4%}'.format(epoch_i, train_accuracy)
            print(content)
            test_accuracy = sess.run(accuracy, feed_dict={features: test_data, labels: test_label})
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)

    # Test trained model
    final_accuracy = sess.run(accuracy, feed_dict={features: test_data, labels: test_label})
    print('---------------------------------------')
    print('accuracy on the test data set= %10.4f' % final_accuracy)

    save_path = '{}weights/'.format(log_path)
    save_parameters(sess, save_path, saver, '{0}-{1}'.format(lr, units))
    write_to_file(log_path + 'accuracy_{}_{}.txt'.format(lr, units), '{}\n{}'.format(train_acc, test_acc), True)

    return final_accuracy


def main():
    classify_time = time.time()

    # Train and test the model.
    write_to_file(log_path + 'accuracy_summary.txt', 'accuracy', True)
    # accuracy = train(5e-6, 2048)
    for lr in LEARNING_RATE:
        for unit in HIDDEN_UNITS:
            test_accuracy = train(lr, unit)
            write_to_file(log_path + 'accuracy_summary.txt',
                          'lr:{0} | units: {1} | test accuracy: {2}\n'.format(lr, unit, test_accuracy), False)
    classified_time = time.time() - classify_time

    print('Classified time: {0:.4f} seconds.'.format(classified_time))


if __name__ == '__main__':
    main()
