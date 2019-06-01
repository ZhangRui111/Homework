import numpy as np
import os


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


def main():
    result = np.genfromtxt('./logs/bert_output/test_results.tsv')
    result_inds = np.argmax(result, axis=1)
    result_list = []

    write_to_csv('./logs/QNLI.tsv', 'index\tprediction\n', True)
    for i in range(len(result_inds)):
        if result_inds[i] == 1:
            result_list.append(str(i) + '\t' + 'entailment' + '\n')
        else:
            result_list.append(str(i) + '\t' + 'not_entailment' + '\n')

    write_to_csv('./logs/QNLI.tsv', result_list, False)


if __name__ == '__main__':
    main()
