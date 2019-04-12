from sklearn import svm, preprocessing
import numpy as np
from sklearn.metrics import accuracy_score

train_data = np.load('../data/train_data.npy')
train_label = np.load('../data/train_label.npy')
test_data = np.load('../data/test_data.npy')
test_label = np.load('../data/test_label.npy')
train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)

from sklearn.utils import shuffle
train_data, train_label = shuffle(train_data, train_label)
train_data = train_data[0: 8000, :]
train_label = train_label[0: 8000]


def one_vs_rest_training(target, label):
    tr_label = []
    for i in range(len(label)):
        if label[i] == target:
            tr_label.append(1)
        else:
            tr_label.append(0)
    return tr_label


def train():
    tr_label_1 = one_vs_rest_training(1, train_label)
    tr_label_2 = one_vs_rest_training(0, train_label)
    tr_label_3 = one_vs_rest_training(-1, train_label)
    print('training 1..')
    clf1 = svm.SVC(kernel='rbf', C=1, probability=True).fit(train_data, tr_label_1)
    print('training 2..')
    clf2 = svm.SVC(kernel='rbf', C=1, probability=True).fit(train_data, tr_label_2)
    print('training 3..')
    clf3 = svm.SVC(kernel='rbf', C=1, probability=True).fit(train_data, tr_label_3)
    return clf1, clf2, clf3


def get_pred(label):
    pred = 0
    loc = np.argmax(label)
    if loc == 0:
        pred = 1
    elif loc == 1:
        pred = 0
    elif loc == 2:
        pred = -1
    return pred


def main():
    clf1, clf2, clf3 = train()
    print('predicting 1..')
    pred1 = clf1.predict_proba(test_data)
    print('predicting 2..')
    pred2 = clf2.predict_proba(test_data)
    print('predicting 3..')
    pred3 = clf3.predict_proba(test_data)
    pred = []

    for i in range(len(pred1)):
        label = [pred1[i][1], pred2[i][1], pred3[i][1]]
        pred.append(get_pred(label))
    print(accuracy_score(pred, test_label))


if __name__ == "__main__":
    main()
