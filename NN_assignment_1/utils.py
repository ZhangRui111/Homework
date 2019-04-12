import csv
import os
import scipy.io as sio



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


def save_parameters(sess, save_path, saver, name):
    """ Save Network's weights.
    """
    exist_or_create_folder(save_path)
    saver.save(sess, '{0}{1}'.format(save_path, name))
    print('save weights')


def read_data(root_path):
    train_data = sio.loadmat('{0}{1}.mat'.format(root_path, 'train_data'))['train_data']  # (499, 310)
    train_label = sio.loadmat('{0}{1}.mat'.format(root_path, 'train_label'))['train_label']  # (499, 1)
    test_data = sio.loadmat('{0}{1}.mat'.format(root_path, 'test_data'))['test_data']  # (343, 310)
    test_label = sio.loadmat('{0}{1}.mat'.format(root_path, 'test_label'))['test_label']  # (343, 1)

    return train_data, train_label, test_data, test_label


def write_to_file(path, content, overWrite):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass
    if overWrite is True:
        with open(path, 'w') as f:
            f.write(content)
    else:
        with open(path, 'a') as f:
            f.write(content)


def write_to_csv(path, data_list, overWrite=True):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass
    if overWrite is True:
        with open(path, 'w') as file:
            for line in data_list:
                file.write(line)
    else:
        with open(path, 'a') as file:
            for line in data_list:
                file.write(line)


def txt2csv(old_path, new_path):
    if not os.path.exists(os.path.dirname(old_path)):
        return 0
    else:
        if not os.path.exists(os.path.dirname(new_path)):
            try:
                os.makedirs(os.path.dirname(new_path))
            except OSError:
                pass
        array = []
        with open(old_path, 'r') as f:
            for line in f:
                array.append(line)

        with open(new_path, 'w') as file:
            for line in array:
                file.write(line)
        return 1
