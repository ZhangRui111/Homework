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


def plot_rate(data, path, index):
    data_average = []
    size = len(data)
    for i in range(25, size):
        data_average.append(sum(data[(i-25):i])/25)

    np.save('{0}data_average_{1}'.format(path, index), np.array(data_average))
    np.save('{0}data_{1}'.format(path, index), np.array(data))

    plt.plot(np.arange(len(data_average)), data_average)
    plt.ylabel('success rate')
    plt.xlabel('episode')
    # plt.show()
    plt.savefig(path + 'success_rate.png')
    plt.close()


def plot_rate_average(basepath, color_bg, color_fg):
    all_data_average = []
    for i in range(15):
        path = '{0}{1}/'.format(basepath, i + 1)
        all_data_average.append(np.load('{0}data_average_{1}.npy'.format(path, i + 1)))
    # all_data_average_np = all_data_average[0]
    # for j in range(9):
    #     all_data_average_np = np.stack((all_data_average_np, all_data_average[j+1]))
    all_data_average_np = np.stack((all_data_average[0], all_data_average[1],
                                    all_data_average[2], all_data_average[3],
                                    all_data_average[4], all_data_average[5],
                                    all_data_average[6], all_data_average[7],
                                    all_data_average[8], all_data_average[9],
                                    all_data_average[10], all_data_average[11],
                                    all_data_average[12], all_data_average[13],
                                    all_data_average[14]))

    all_average_data = np.mean(all_data_average_np, axis=0)
    all_std = np.std(all_data_average_np, axis=0)
    np.save('{0}mean_result'.format(basepath), all_average_data)
    # print(all_average_data)
    print('{0} -- {1}'.format(basepath.split('/')[-2], np.sum(all_std)))

    for i in range(15):
        plt.plot(np.arange(len(all_data_average[i])), all_data_average[i], c=color_bg)

    plt.plot(np.arange(len(all_average_data)), all_average_data, c=color_fg, label=basepath.split('/')[-2])
    plt.ylabel('success rate')
    plt.xlabel('episode')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('{0}success_rate.png'.format(basepath))
    plt.close()


def plot_cmp(clip=False):
    dqn_line = np.load('./logs/backup/dqn/mean_result.npy')
    double_dqn_line = np.load('./logs/backup/double_dqn/mean_result.npy')
    dueling_dqn_line = np.load('./logs/backup/dueling_dqn/mean_result.npy')

    # print('dqn -- {}'.format(np.mean(dqn_line[300:])))
    # print('double dqn -- {}'.format(np.mean(double_dqn_line[300:])))
    # print('dueling dqn -- {}'.format(np.mean(dueling_dqn_line[300:])))

    if clip:
        plt.plot(np.arange(len(dqn_line[:200])), dqn_line[:200], c='xkcd:black', label='dqn')
        plt.plot(np.arange(len(double_dqn_line[:200])), double_dqn_line[:200], c='xkcd:green', label='double_dqn')
        plt.plot(np.arange(len(dueling_dqn_line[:200])), dueling_dqn_line[:200], c='xkcd:red', label='dueling_dqn')
    else:
        plt.plot(np.arange(len(dqn_line)), dqn_line, c='xkcd:black', label='dqn')
        plt.plot(np.arange(len(double_dqn_line)), double_dqn_line, c='xkcd:green', label='double_dqn')
        plt.plot(np.arange(len(dueling_dqn_line)), dueling_dqn_line, c='xkcd:red', label='dueling_dqn')
    plt.ylabel('success rate')
    plt.xlabel('episode')
    plt.legend(loc='best')
    # plt.show()
    if clip:
        plt.savefig('./logs/backup/success_rate_clip.png')
    else:
        plt.savefig('./logs/backup/success_rate.png')
    plt.close()


def main():
    # plot_rate_average('./logs/backup/dqn/', 'xkcd:silver', 'xkcd:black')
    # plot_rate_average('./logs/backup/double_dqn/', 'xkcd:yellowgreen', 'xkcd:green')
    # plot_rate_average('./logs/backup/dueling_dqn/', 'xkcd:gold', 'xkcd:red')
    plot_cmp(clip=True)


if __name__ == '__main__':
    main()
