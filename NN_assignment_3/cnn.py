import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mn
import time

from utils import exist_or_create_folder, plot_accuracy, write_to_file

NUM_CLASS = 10
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
log_path = './logs/cnn/'
weights_path = './logs/cnn/weights/lenet'


def build_network(sess):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None,))
    one_hot_y = tf.one_hot(y, NUM_CLASS)

    # Hyperparameters when initialize the weights.
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b)

    # Layer 2: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.relu(tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b)

    # PLayer 4: ooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = tf.contrib.layers.flatten(pool_2)

    # Layer 5: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.nn.relu(tf.matmul(fc1, fc1_w) + fc1_b)

    # Layer 6: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)

    # Layer 7: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, NUM_CLASS), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(NUM_CLASS))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    return [[x, y, one_hot_y],
            [logits, loss, train_op, correct_prediction, accuracy],
            [conv1, pool_1, conv2, pool_2, fc1, fc2, logits]]


def main():
    mnist = mn.input_data.read_data_sets("./data/", one_hot=False, reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_valid, y_valid = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    classify_time = time.time()

    # Pad images with 0s
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=(0, 0))
    X_valid = np.pad(X_valid, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=(0, 0))
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=(0, 0))

    # # Show a training sample.
    # index = random.randint(0, len(X_train))
    # image = X_train[index].squeeze()
    # plt.figure(figsize=(1, 1))
    # plt.imshow(image, cmap="gray")
    # plt.show()
    # print(y_train[index])

    # Shuffle the training data.
    X_train, y_train = shuffle(X_train, y_train)

    exist_or_create_folder(weights_path)

    accuracy_valid = []
    with tf.Session() as sess:
        # build LeNet.
        net = build_network(sess)
        saver = tf.train.Saver()
        x, y, one_hot_y = net[0]
        logits, loss, train_op, correct_prediction, accuracy_op = net[1]

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
    plot_accuracy(accuracy_valid, log_path, 'Convolutional NN')

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

        # # Visualize the deep features.
        index = random.randint(0, len(X_train))
        image = X_train[index].squeeze()
        plt.figure(figsize=(1, 1))
        plt.imshow(image, cmap="gray")
        plt.savefig('./logs/cnn/image.png')
        print(y_train[index])

        conv1, pool_1, conv2, pool_2, fc1, fc2, logits = net[2]
        v_conv1, v_pool_1, v_conv2, v_pool_2, v_fc1, v_fc2, v_logits = sess.run(
            [conv1, pool_1, conv2, pool_2, fc1, fc2, logits], feed_dict={x: X_train[index][np.newaxis, :], y: np.array([y_train[index]])})
        image_conv1 = v_conv1.squeeze()
        image_pool_1 = v_pool_1.squeeze()
        image_conv2 = v_conv2.squeeze()
        image_pool_2 = v_pool_2.squeeze()
        image_fc1 = v_fc1
        image_fc2 = v_fc2
        image_logits = v_logits.squeeze()
        fig = plt.figure(figsize=(8, 12))
        fig.suptitle('Layer 1: convolutional (28*28*6)')
        for i in range(0, 6):
            fig.add_subplot(2, 3, i+1)
            plt.imshow(image_conv1[:, :, i], cmap="gray")
        plt.savefig('./logs/cnn/ly1_conv.png')
        fig = plt.figure(figsize=(8, 12))
        fig.suptitle('layer 2: pooling (14*14*6)')
        for i in range(0, 6):
            fig.add_subplot(2, 3, i+1)
            plt.imshow(image_pool_1[:, :, i], cmap="gray")
        plt.savefig('./logs/cnn/ly2_pool.png')
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('layer 3: convolutional (10*10*16)')
        for i in range(0, 16):
            fig.add_subplot(4, 4, i+1)
            plt.imshow(image_conv2[:, :, i], cmap="gray")
        plt.savefig('./logs/cnn/ly3_conv.png')
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('layer 4: pooling (5*5*16)')
        for i in range(0, 16):
            fig.add_subplot(4, 4, i+1)
            plt.imshow(image_pool_2[:, :, i], cmap="gray")
        plt.savefig('./logs/cnn/ly4_pool.png')
        fig = plt.figure(figsize=(15, 2))
        fig.suptitle('layer 5: fully-connected (1*120)')
        plt.imshow(image_fc1, cmap="gray")
        plt.savefig('./logs/cnn/ly5_fc.png')
        fig = plt.figure(figsize=(15, 2))
        fig.suptitle('layer 6: fully-connected (1*84)')
        plt.imshow(image_fc2, cmap="gray")
        plt.savefig('./logs/cnn/ly6_fc.png')
        fig = plt.figure(figsize=(15, 2))
        fig.suptitle('layer 7: fully-connected (1*10)')
        plt.imshow(image_logits[np.newaxis, :], cmap="gray")
        plt.savefig('./logs/cnn/ly7_output.png')

    classified_time = time.time() - classify_time
    print('Classified time: {0:.4f} seconds.'.format(classified_time))
    write_to_file('{0}time.txt'.format(log_path), classified_time, True)


if __name__ == '__main__':
    main()

