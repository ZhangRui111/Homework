import numpy as np


def one_vs_rest_label(label, num_class=3):
    labels = []
    for cls in range(num_class):
        inds = np.where(np.equal(label, cls))[0]
        cls_label = np.zeros(label.shape, dtype=np.int32)
        cls_label[inds] = 1
        labels.append(cls_label)

    return labels


def min_max_data_label(data, label, num_class=3):
    labels = one_vs_rest_label(label, num_class)
    min_max_labels = []
    min_max_datas = []
    for cls in range(num_class):
        pos_inds = np.where(np.equal(labels[cls], 1))[0]
        neg_inds = np.where(np.equal(labels[cls], 0))[0]
        np.random.shuffle(neg_inds)
        num_neg = neg_inds.shape[0]
        split_ind = int(num_neg / 2)
        sub_label = []
        sub_label.append(np.concatenate((labels[cls][pos_inds], labels[cls][neg_inds[0:split_ind]])))
        sub_label.append(np.concatenate((labels[cls][pos_inds], labels[cls][neg_inds[split_ind:]])))
        sub_data = []
        sub_data.append(np.concatenate((data[pos_inds], data[neg_inds[0:split_ind]])))
        sub_data.append(np.concatenate((data[pos_inds], data[neg_inds[split_ind:]])))

        min_max_labels.append(sub_label)
        min_max_datas.append(sub_data)

    return min_max_datas, min_max_labels
