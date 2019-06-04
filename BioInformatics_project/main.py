import numpy as np
import tensorflow as tf

from utils import read_txt_lines

DNN = False  # Whether use deep neural network for classification.


def run_dnn_classifier():
    pass


def run_svm_classifier():
    pass


def main():
    # process the data.
    genes_path = './data/microarray.original.txt'
    labels_path = './data/E-TABM-185.sdrf.txt'

    # run the classifier.
    if DNN:
        run_dnn_classifier()
    else:
        run_svm_classifier()


if __name__ == '__main__':
    main()
