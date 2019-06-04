import numpy as np
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


def read_txt_full(file_path):
    with open(file_path, 'r') as f:
        read_data = f.read()
        return read_data


def read_txt_line(file_path, limitation=1):
    read_data_list = []
    assert type(limitation) == int, "The second parameter limitation must be int type."
    with open(file_path, 'r') as f:
        for i in range(limitation):
            read_data_list.append(f.readline())
        return read_data_list


def read_txt_lines(file_path):
    with open(file_path, 'r') as f:
        read_data = f.readlines()
        return read_data


def process_tsv(data_list):
    size = len(data_list)
    processed_data_list = []
    for i in range(size):
        line = data_list[i].strip('\n')
        processed_data_list.append(line.split('\t'))
    return processed_data_list


def process_genes(data_list):
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
    pass


def distinguish_labels(data_list):
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


def write_txt_lines(file_path, data_list, overwrite=False):
    size = len(data_list)
    if overwrite:
        mode = 'w'
    else:
        mode = 'a'
    # for i in range(size):
    #     with open(file_path, mode) as f:
    #         for item in data_list[i]:
    #             f.write("{0}\n".format(str(item)))
    with open(file_path, mode) as f:
        for i in range(size):
            f.write("{0}\n".format(str(data_list[i])))


def main():
    # data = read_txt_line('./data/raw_data/microarray.original.txt', limitation=10)
    # data = read_txt_lines('./data/raw_data/microarray.original.txt')
    # data = read_txt_line('./data/raw_data/E-TABM-185.sdrf.txt', limitation=100)
    data = read_txt_lines('./data/raw_data/E-TABM-185.sdrf.txt')
    data = process_tsv(data)
    # process_genes(data)
    # distinguish_labels(data)


if __name__ == '__main__':
    main()
