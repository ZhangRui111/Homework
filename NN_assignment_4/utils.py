import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat


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


def plot_accuracy(list, save_path, label):
    size = len(list)
    x_axis_data = np.arange(0, size)
    plt.plot(x_axis_data, 100 * np.asarray(list), label=label)
    plt.title('accuracy on the test set')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/%')
    y_axis_ticks = [90, 92, 94, 96, 98, 100]
    plt.yticks(y_axis_ticks)
    for items in y_axis_ticks:
        plt.hlines(items, x_axis_data.min(), x_axis_data.max(), colors="#D3D3D3", linestyles="dashed")
    plt.legend(loc='best')
    if save_path is not None:
        plt.savefig(save_path + 'accuracy.png')
    # plt.show()


def read_data(x_data_path, y_data_path):
    """ Read *.mat data """
    x = loadmat(x_data_path)['X'][0]
    y = loadmat(y_data_path)['Y'][0]
    return x, y
