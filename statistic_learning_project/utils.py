import csv
import os


def read_test(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        test_data = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                test_data.append([float(row[i]) for i in range(4097)])
                line_count += 1
        return test_data


def read_train(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        train_data = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                train_data.append([float(row[i]) for i in range(4098)])
                line_count += 1
        return train_data


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
