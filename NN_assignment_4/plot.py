import numpy as np
import matplotlib.pyplot as plt

from utils import exist_or_create_folder


def main():
    ff_accuracy = np.loadtxt('./logs/ff/accuracy.out', dtype=float)
    out_accuracy = np.loadtxt('./logs/cnn/accuracy.out', dtype=float)
    size = len(ff_accuracy)
    x_axis_data = np.arange(0, size)
    plt.plot(x_axis_data, 100 * ff_accuracy, label='feed forward NN')
    plt.plot(x_axis_data, 100 * out_accuracy, label='Convolutional NN')
    plt.title('accuracy on the test set')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/%')
    y_axis_ticks = [90, 92, 94, 96, 98, 100]
    plt.yticks(y_axis_ticks)
    for items in y_axis_ticks:
        plt.hlines(items, x_axis_data.min(), x_axis_data.max(), colors="#D3D3D3", linestyles="dashed")
    plt.legend(loc='best')
    exist_or_create_folder('./logs/')
    plt.savefig('./logs/accuracy_cmp.png')
    plt.show()


if __name__ == '__main__':
    main()
