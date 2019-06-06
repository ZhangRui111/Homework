import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA


def exist_or_create_folder(path):
    """ Check if the path exists, if not exists, create it. """
    flag = False
    pure_path = os.path.dirname(path)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def read_txt_full(file_path):
    """ Read the full file. """
    with open(file_path, 'r') as f:
        read_data = f.read()
        return read_data


def read_txt_line(file_path, limitation=1):
    """ Read some lines of the file from the beginning."""
    read_data_list = []
    assert type(limitation) == int, "The second parameter limitation must be int type."
    with open(file_path, 'r') as f:
        for i in range(limitation):
            read_data_list.append(f.readline())
        return read_data_list


def read_txt_lines(file_path):
    """ Read the full file and return the content in a list involving all lines. """
    with open(file_path, 'r') as f:
        read_data = f.readlines()
        return read_data


def process_tsv(data_list):
    """ Separate every line by '\t'. """
    size = len(data_list)
    processed_data_list = []
    for i in range(size):
        line = data_list[i].strip('\n')
        processed_data_list.append(line.split('\t'))
    return processed_data_list


def process_genes(data_list):
    """
    Process the gene features and convert into a single
    numpy array of shape (5896, 22283), then, save it to genes.out.

    """
    data_list2float = []
    data_list.pop(0)
    size = len(data_list)
    for i in range(size):
        data_list[i].pop(0)
        data_list2float.append(np.array(list(map(float, data_list[i]))))
    data_list2float2np = np.transpose(np.array(data_list2float))
    print(data_list2float2np.shape)
    np.savetxt('./data/genes.out', data_list2float2np, fmt='%.9f')


def process_labels(data_list, index):
    """ Process one specific label and convert to int type. """
    data_list.pop(0)
    rows = len(data_list)
    selected_labels_list = []
    for i in range(rows):
        selected_labels_list.append(develop_stage2int(data_list[i][index]))
    np_labels = np.asarray(selected_labels_list)[:, np.newaxis]
    np.savetxt('./data/labels.out', np_labels, fmt='%d')

    return np_labels


def distinguish_labels(data_list):
    """
    Distinguish specific items for each label,
    then, save it to distinguished_labels.txt.
    """
    data_list_reverse = []
    data_list.pop(0)
    rows = len(data_list)
    columes = len(data_list[1])
    for j in range(columes):
        item = []
        for i in range(rows):
            item.append(data_list[i][j])
            item = list(dict.fromkeys(item))  # remove duplicates from a List.
        data_list_reverse.append(item)
    data_list_reverse.pop(0)
    data_list_reverse.pop(10)
    write_txt_lines('./data/distinguished_labels', data_list_reverse, overwrite=True)

    return data_list


def develop_stage2int(label):
    """ Convert the label 'Characteristics [DevelopmentalStage]' to int type. """
    # ['  ', 'adult', 'embryo', 'fetus', 'embryoid_bodies', 'newborn', 'embryonic', 'fetal', 'infant']
    if label == 'adult':
        int_label = 0
    elif label == 'embryo':
        int_label = 1
    elif label == 'fetus':
        int_label = 2
    elif label == 'embryoid_bodies':
        int_label = 3
    elif label == 'newborn':
        int_label = 4
    elif label == 'embryonic':
        int_label = 5
    elif label == 'fetal':
        int_label = 6
    elif label == 'infant':
        int_label = 7
    else:
        int_label = -1

    return int_label


def write_txt_lines(file_path, data_list, overwrite=False):
    """ Write the content by lines. """
    size = len(data_list)
    if overwrite:
        mode = 'w'
    else:
        mode = 'a'
    # # list in list.
    # for i in range(size):
    #     with open(file_path, mode) as f:
    #         for item in data_list[i]:
    #             f.write("{0}\n".format(str(item)))
    # # item in list.
    with open(file_path, mode) as f:
        for i in range(size):
            f.write("{0}\n".format(str(data_list[i])))


def pre_processing():
    """
    Final processing of the raw data to divide the
    train_genes, train_labels, test_genes, test_labels
    and save them to files.
    """
    genes = np.loadtxt('./data/genes.out')
    # genes = np.zeros((10, 200))
    print(genes.shape)
    labels = np.loadtxt('./data/labels.out', dtype=int)[:, np.newaxis]
    # labels = np.ones((10, 1))
    print(labels.shape)
    inds = np.where(labels < 0)[0]

    # Remove those genes that don't have labels.
    dataset = np.concatenate((genes, labels), axis=1)
    pure_dataset = np.delete(dataset, inds, axis=0)
    print(pure_dataset.shape)

    # Shuffle the pure_dataset along the first axis.
    np.random.shuffle(pure_dataset)

    # Divide the pure_dataset.
    size = pure_dataset.shape[0]
    train_size = int(0.8 * size)

    train_genes = pure_dataset[:train_size, :-1]
    train_labels = pure_dataset[:train_size, -1]
    test_genes = pure_dataset[train_size:, :-1]
    test_labels = pure_dataset[train_size:, -1]
    np.savetxt('./data/train_genes.out', train_genes, fmt='%.9f')
    np.savetxt('./data/train_labels.out', train_labels, fmt='%d')
    np.savetxt('./data/test_genes.out', test_genes, fmt='%.9f')
    np.savetxt('./data/test_labels.out', test_labels, fmt='%d')

    return train_genes, train_labels, test_genes, test_labels


def load_data_from_files():
    """
    Load pre-processed data from files.
    Called after pre_processing().
    """
    train_genes = np.loadtxt('./data/train_genes.out')
    train_labels = np.loadtxt('./data/train_labels.out', dtype=int)[:, np.newaxis]
    test_genes = np.loadtxt('./data/test_genes.out')
    test_labels = np.loadtxt('./data/test_labels.out', dtype=int)[:, np.newaxis]
    print(train_genes.shape)
    print(train_labels.shape)
    print(test_genes.shape)
    print(test_labels.shape)

    return train_genes, train_labels, test_genes, test_labels


def plot_pca_2D(X, y):
    """ Plot the pca result in 2D. """
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    # # Colors, Markers and Legends.
    colors = {0: '#FF5733',
              1: '#FCFF33',
              2: '#4CFC1C',
              3: '#09FEE2',
              4: '#D912D9',
              5: '#000000',
              6: '#125FD9',
              7: '#2F9A5F',
              8: '#CBFF33',
              9: '#FFA833'}

    markers = {0: '.',
               1: 'v',
               2: '^',
               3: '<',
               4: '>',
               5: 's',
               6: 'p',
               7: 'P',
               8: '+',
               9: 'x'}

    legends = {0: 'adult',
               1: 'embryo',
               2: 'fetus',
               3: 'embryoid_bodies',
               4: 'newborn',
               5: 'embryonic',
               6: 'fetal',
               7: 'infant',
               8: 'null',
               9: 'null'}

    data = []
    for lab in range(8):
        trace = dict(
            label=lab,
            data=X[y == lab, :]
        )
        data.append(trace)

    for i in range(len(data)):
        data_x = data[i]['data'][:, 0].ravel()
        data_y = data[i]['data'][:, 1].ravel()
        plt.scatter(data_x, data_y, c=colors[data[i]['label']], label=legends[data[i]['label']],
                    marker=markers[data[i]['label']])

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='best')
    plt.show()


def plot_pca_3D(X, y):
    """ Plot the pca result in 3D. """
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    # # Colors, Markers and Legends.
    colors = {0: '#FF5733',
              1: '#FCFF33',
              2: '#4CFC1C',
              3: '#09FEE2',
              4: '#D912D9',
              5: '#000000',
              6: '#125FD9',
              7: '#2F9A5F',
              8: '#CBFF33',
              9: '#FFA833'}

    markers = {0: '.',
               1: 'v',
               2: '^',
               3: '<',
               4: '>',
               5: 's',
               6: 'p',
               7: 'P',
               8: '+',
               9: 'x'}

    legends = {0: 'adult',
               1: 'embryo',
               2: 'fetus',
               3: 'embryoid_bodies',
               4: 'newborn',
               5: 'embryonic',
               6: 'fetal',
               7: 'infant',
               8: 'null',
               9: 'null'}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = []
    for lab in range(8):
        # if lab == 0:
        #     continue
        trace = dict(
            label=lab,
            data=X[y == lab, :]
        )
        data.append(trace)

    for i in range(len(data)):
        label = data[i]['label']
        data_x = data[i]['data'][:, 0].ravel()
        data_y = data[i]['data'][:, 1].ravel()
        data_z = data[i]['data'][:, 2].ravel()
        ax.scatter(data_x, data_y, data_z, c=colors[label], label=legends[label], marker=markers[label])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend(loc='best')
    plt.show()


def sklearn_pca(keep_dimens=500):
    """
    Apply PCA to train_genes, train_labels, test_genes,
    test_labels and save them to files.
    """
    train_genes, train_labels, test_genes, test_labels = load_data_from_files()

    size_train = train_genes.shape[0]
    all_genes = np.concatenate((train_genes, test_genes), axis=0)

    all_genes_std = StandardScaler().fit_transform(all_genes)
    all_genes_sklearn_pca = sklearnPCA(n_components=keep_dimens)
    all_genes_pca = all_genes_sklearn_pca.fit_transform(all_genes_std)

    all_labels = np.vstack((train_labels, test_labels)).ravel()

    if keep_dimens == 2:
        plot_pca_2D(all_genes_pca, all_labels)

    if keep_dimens == 3:
        plot_pca_3D(all_genes_pca, all_labels)

    train_genes_pca = all_genes_pca[:size_train, :]
    test_genes_pca = all_genes_pca[size_train:, :]

    np.savetxt('./data/pca_data/train_genes_pca_{}.out'.format(keep_dimens),
               train_genes_pca, fmt='%.9f')
    np.savetxt('./data/pca_data/train_labels_pca.out', train_labels, fmt='%d')
    np.savetxt('./data/pca_data/test_genes_pca_{}.out'.format(keep_dimens),
               test_genes_pca, fmt='%.9f')
    np.savetxt('./data/pca_data/test_labels_pca.out', test_labels, fmt='%d')

    # return train_genes_pca, train_labels_pca, test_genes_pca, test_labels_pca
    return train_genes_pca, test_genes_pca


def load_pca_data_from_files(keep_dimens=500):
    """ Load pca data from files. Called after sklearn_pca(). """
    train_genes_pca = np.loadtxt('./data/pca_data/train_genes_pca_{}.out'.format(keep_dimens))
    train_labels_pca = np.loadtxt('./data/pca_data/train_labels.out', dtype=int)
    test_genes_pca = np.loadtxt('./data/pca_data/test_genes_pca_{}.out'.format(keep_dimens))
    test_labels_pca = np.loadtxt('./data/pca_data/test_labels.out', dtype=int)
    print(train_genes_pca.shape)
    print(train_labels_pca.shape)
    print(test_genes_pca.shape)
    print(test_labels_pca.shape)

    return train_genes_pca, train_labels_pca, test_genes_pca, test_labels_pca


# def main():
    # Data processing.
    # data = read_txt_line('./data/raw_data/microarray.original.txt', limitation=10)
    # data = read_txt_lines('./data/raw_data/microarray.original.txt')
    # data = read_txt_line('./data/raw_data/E-TABM-185.sdrf.txt', limitation=100)
    # data = read_txt_lines('./data/raw_data/E-TABM-185.sdrf.txt')
    # data = process_tsv(data)
    # process_genes(data)
    # distinguish_labels(data)
    # process_labels(data, index=5)
    # pre_processing()

    # Apply PCA to train_genes, train_labels, test_genes, test_labels.
    # sklearn_pca(keep_dimens=800)
    # train_genes, train_labels, test_genes, test_labels = load_pca_data_from_files(keep_dimens=600)
    # print(train_genes.shape)
    # print(train_labels.shape)
    # print(test_genes.shape)
    # print(test_labels.shape)


# if __name__ == '__main__':
#     main()
