import numpy as np
import os


def exist_or_create_folder(path_name):
    """
    Check whether a path exists, if not, then create this path.
    :param path_name: i.e., './logs/log.txt' or './logs/'
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    pure_path = os.path.dirname(path_name)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def write_to_file(path, content, overWrite):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass
    if overWrite is True:
        with open(path, 'w') as f:
            f.write(str(content))
    else:
        with open(path, 'a') as f:
            f.write(str(content))


def one_vs_rest_labels(label, num_class=3):
    """ One verses rest labels (target class label = 1, others label = 0) """
    labels = []
    for cls in range(num_class):
        inds = np.where(np.equal(label, cls))[0]
        cls_label = np.zeros(label.shape, dtype=np.int32)
        cls_label[inds] = 1
        labels.append(cls_label)

    return labels


def min_max_labels_random(data, label, num_class=3):
    """ Min-Max-Module SVM and decompose task randomly. """
    labels = one_vs_rest_labels(label, num_class)
    mm_datas = []
    mm_labels = []
    for cls in range(num_class):
        pos_inds = np.where(np.equal(labels[cls], 1))[0]
        neg_inds = np.where(np.equal(labels[cls], 0))[0]
        np.random.shuffle(neg_inds)
        num_neg = neg_inds.shape[0]
        split_ind = int(num_neg / 2)
        sub_label = [np.concatenate((labels[cls][pos_inds], labels[cls][neg_inds[0:split_ind]])),
                     np.concatenate((labels[cls][pos_inds], labels[cls][neg_inds[split_ind:]]))]
        sub_data = [np.concatenate((data[pos_inds], data[neg_inds[0:split_ind]])),
                    np.concatenate((data[pos_inds], data[neg_inds[split_ind:]]))]

        mm_labels.append(sub_label)
        mm_datas.append(sub_data)

    return mm_datas, mm_labels


def min_max_labels_prior(data, label, num_class=3):
    """ Min-Max-Module SVM and decompose task randomly with prior knowledge. """
    one_inds = np.where(np.equal(label, 0))[0]
    two_inds = np.where(np.equal(label, 1))[0]
    three_inds = np.where(np.equal(label, 2))[0]
    inds = [one_inds, two_inds, three_inds]

    mm_datas = []
    mm_labels = []
    for cls in range(num_class):
        inds_all = [0, 1, 2]
        inds_all.remove(cls)
        sub_label = [np.concatenate((label[inds[cls]], label[inds[inds_all[0]]])),
                     np.concatenate((label[inds[cls]], label[inds[inds_all[1]]]))]
        sub_data = [np.concatenate((data[inds[cls]], data[inds[inds_all[0]]])),
                    np.concatenate((data[inds[cls]], data[inds[inds_all[1]]]))]

        mm_labels.append(sub_label)
        mm_datas.append(sub_data)

    return mm_datas, mm_labels
