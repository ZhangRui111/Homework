import numpy as np
import matplotlib.pyplot as plt
import os


def exist_or_create_folder(path_name):
    flag = False
    pure_path = os.path.dirname(path_name)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def write_to_file(file_path, content, overwrite=False):
    exist_or_create_folder(file_path)
    if overwrite is True:
        with open(file_path, 'w') as f:
            f.write(str(content))
    else:
        with open(file_path, 'a') as f:
            f.write(str(content))


def plot_cost(data, path):
    data_average = []
    size = len(data)
    for i in range(50, size):
        data_average.append(sum(data[(i-50):i])/50)

    np.save('./logs/dqn/data_average_rate.out', np.array(data_average))
    np.save('./logs/dqn/data_rate.out', np.array(data))

    plt.plot(np.arange(len(data_average)), data_average)
    plt.ylabel('success rate')
    plt.xlabel('episode')
    # plt.show()
    plt.savefig(path)
    plt.close()


def plot_rate(data, path):
    data_average = []
    size = len(data)
    for i in range(50, size):
        data_average.append(sum(data[(i-50):i])/50)

    np.save('./logs/dqn/data_average.out', np.array(data_average))
    np.save('./logs/dqn/data.out', np.array(data))

    plt.plot(np.arange(len(data_average)), data_average)
    plt.ylabel('episode steps')
    plt.xlabel('episode')
    # plt.show()
    plt.savefig(path)
    plt.close()