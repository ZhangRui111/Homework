import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mn
import time

from utils import exist_or_create_folder, plot_accuracy, write_to_file

FEATURE_SIZE = 28*28
NUM_CLASS = 10
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
log_path = './logs/ff/'
weights_path = './logs/ff/weights/lenet'


def build_network(sess):
    # Placeholders
    features = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    labels = tf.placeholder(tf.float32, [None, NUM_CLASS])

    with tf.variable_scope('eval_net'):
        hl = tf.contrib.layers.fully_connected(features, 512, activation_fn=tf.nn.relu)
        out_softmax = tf.contrib.layers.fully_connected(hl, NUM_CLASS, activation_fn=tf.nn.softmax)

    with tf.variable_scope('loss'):
        loss = -tf.reduce_sum(labels * tf.log(out_softmax))

    with tf.variable_scope('train_accuracy'):
        # train.
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        # accuracy evaluation.
        correct_prediction = tf.equal(tf.argmax(out_softmax, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    return [[features, labels], [out_softmax, loss, train_op, accuracy]]


def main():
    mnist = mn.input_data.read_data_sets("./data/", one_hot=True, reshape=True)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_valid, y_valid = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    classify_time = time.time()

    # Shuffle the training data.
    X_train, y_train = shuffle(X_train, y_train)

    exist_or_create_folder(weights_path)

    accuracy_valid = []
    with tf.Session() as sess:
        # build LeNet.
        net = build_network(sess)
        saver = tf.train.Saver()
        x, y = net[0]
        out_softmax, loss, train_op, accuracy_op = net[1]

        num_examples = len(X_train)
        print('Training...')
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            # Batch gradient descent.
            n_batches = num_examples // BATCH_SIZE
            for ind in range(0, n_batches):
                start = ind * BATCH_SIZE
                end = (ind + 1) * BATCH_SIZE
                batch_x, batch_y = X_train[start:end], y_train[start:end]
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            # evaluate accuracy on the validation set.
            num_examples = len(X_valid)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x, batch_y = X_valid[offset:offset + BATCH_SIZE], y_valid[offset:offset + BATCH_SIZE]
                accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
            validation_accuracy = total_accuracy / num_examples

            accuracy_valid.append(validation_accuracy)

            print('EPOCH {0} | Validation Accuracy = {1:.3f}'.format(i + 1, validation_accuracy))

        saver.save(sess, weights_path)

    np.savetxt('{0}accuracy.out'.format(log_path), np.asarray(accuracy_valid), '%f')
    plot_accuracy(accuracy_valid, log_path, 'feed forward NN')

    with tf.Session() as sess:
        saver.restore(sess, weights_path)
        # evaluate accuracy on the test set.
        num_examples = len(X_test)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_test[offset:offset + BATCH_SIZE], y_test[offset:offset + BATCH_SIZE]
            accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        test_accuracy = total_accuracy / num_examples
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    classified_time = time.time() - classify_time
    print('Classified time: {0:.4f} seconds.'.format(classified_time))
    write_to_file('{0}time.txt'.format(log_path), classified_time, True)


if __name__ == '__main__':
    main()

