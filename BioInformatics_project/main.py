import numpy as np
import tensorflow as tf

from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from utils import load_pca_data_from_files

DNN = True  # Whether use deep neural network for classification.
LAMBDA = [0.01, 0.04, 0.08, 0.1, 0.5, 1]  # regularization coefficient.
LEARNING_RATE = 0.0005
MAX_EPISODES = 2000


def build_model(learning_rate, feature_size, num_class, regu_factor, if_nm=False):
    # hidden units
    first_hu = 64
    second_hu = 64
    third_hu = 64

    # Placeholders
    features = tf.placeholder(tf.float32, [None, feature_size])
    labels = tf.placeholder(tf.float32, [None, num_class])

    # 1st layer.
    with tf.name_scope('1st_layer'):
        w1 = tf.Variable(tf.random_normal([feature_size, first_hu], mean=0.0, stddev=0.01))
        b1 = tf.Variable(tf.zeros([first_hu]))
        o1 = tf.nn.relu(tf.matmul(features, w1) + b1)

    # 2nd layer.
    with tf.name_scope('2nd_layer'):
        w2 = tf.Variable(tf.random_normal([first_hu, second_hu], mean=0.0, stddev=0.01))
        b2 = tf.Variable(tf.zeros([second_hu]))
        o2 = tf.nn.relu(tf.matmul(o1, w2) + b2)

    # 3rd layer.
    with tf.name_scope('3rd_layer'):
        w3 = tf.Variable(tf.random_normal([second_hu, third_hu], mean=0.0, stddev=0.01))
        b3 = tf.Variable(tf.zeros([third_hu]))
        o3 = tf.nn.relu(tf.matmul(o2, w3) + b3)

    # 4th layer: softmax predictions.
    with tf.name_scope('4th_layer'):
        w4 = tf.Variable(tf.random_normal([third_hu, num_class], mean=0.0, stddev=0.01))
        b4 = tf.Variable(tf.zeros([num_class]))
        y_pred = tf.matmul(o3, w4) + b4
        y_pred_softmax = tf.nn.softmax(y_pred)  # for prediction

    # loss function and train step
    with tf.name_scope('loss_train'):
        if if_nm is True:
            loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(y_pred_softmax, 1e-10, 1.0))) + \
                   regu_factor * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4))
        else:
            loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(y_pred_softmax, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        # accuracy evaluation
        correct_prediction = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return features, labels, y_pred_softmax, train_step, loss, accuracy


def train(train_features, train_labels, test_features, test_labels, selected_lambda):
    input_size = train_features.shape[1]
    n_classes = train_labels.max() + 1

    features, labels, y_pred_softmax, train_step, loss, accuracy = \
        build_model(learning_rate=LEARNING_RATE,
                    feature_size=input_size,
                    num_class=n_classes,
                    regu_factor=selected_lambda,
                    if_nm=True)

    with tf.Session() as sess:
        tf.summary.FileWriter('./logs', sess.graph)
        sess.run(tf.global_variables_initializer())

        batch_xs_train, batch_ys_train = train_features, np.eye(n_classes)[train_labels]
        batch_xs_test, batch_ys_test = test_features, np.eye(n_classes)[test_labels]
        for episode_i in range(MAX_EPISODES):
            _, loss_step = sess.run([train_step, loss], feed_dict={features: batch_xs_train, labels: batch_ys_train})
            # print(episode_i, ' | ', loss_step)

            # if episode_i % 200 == 0:
            #     train_accuracy = sess.run(accuracy, feed_dict={features: batch_xs_test, labels: batch_ys_test})
            #     str_i = 'step, loss, accuracy = {0} | {1:.4f} | {2:.4%}'.format(episode_i, loss_step, train_accuracy)
            #     print(str_i)

        # Test trained model
        final_accuracy = sess.run(accuracy, feed_dict={features: batch_xs_test, labels: batch_ys_test})
        # print('final accuracy = %10.4f' % final_accuracy)

        return final_accuracy


def run_dnn_classifier(train_genes, train_labels, test_genes, test_labels):
    """ DNN classifier with L2 regularization. """
    for item in LAMBDA:
        accuracy = train(train_genes, train_labels, test_genes, test_labels, item)
        print("Accuracy with lambda={0} is {1}".format(item, accuracy))

    # accuracy = train(train_genes, train_labels, test_genes, test_labels, 0.05)
    # print("Accuracy with lambda={0} is {1}".format(0.05, accuracy))


def run_svm_classifier(train_genes, train_labels, test_genes, test_labels):
    """ SVM classifier. """
    # num_cls = train_labels.max() + 1  # number of classes.

    # # one-vs-one multi-class SVM
    clf = OneVsOneClassifier(svm.SVC(kernel='linear', C=2)).fit(train_genes, train_labels)
    # clf = OneVsOneClassifier(svm.SVC(kernel='rbf', C=2)).fit(train_genes, train_labels)
    # # one-vs-rest multi-class SVM
    # clf = OneVsRestClassifier(svm.SVC(kernel='linear', C=2)).fit(train_genes, train_labels)
    # clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=2)).fit(train_genes, train_labels)
    cls_predict = clf.predict(test_genes)
    n_accuracy = np.where(np.equal(cls_predict, test_labels))[0]
    accuracy = n_accuracy.shape[0] / test_labels.shape[0]
    print('final accuracy {}'.format(accuracy))


def main():
    # Load the processed data.
    train_genes, train_labels, test_genes, test_labels = load_pca_data_from_files(keep_dimens=10)

    # run the classifier.
    if DNN:
        run_dnn_classifier(train_genes, train_labels, test_genes, test_labels)
    else:
        run_svm_classifier(train_genes, train_labels, test_genes, test_labels)


if __name__ == '__main__':
    main()
