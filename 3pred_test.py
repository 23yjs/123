import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """
    读取预测数据
    """
    y_gjs = np.load('data/y_gjs.npy')
    y_hg = np.load('data/y_hg.npy')
    y_ys = np.load('data/y_ys.npy')
    test_y_gjs = np.load('data/test_y_gjs.npy')
    test_y_hg = np.load('data/test_y_hg.npy')
    test_y_ys = np.load('data/test_y_ys.npy')

    """
    画图展示
    """
    plt.plot(test_y_ys, color='red', label='Original')
    plt.plot(y_ys, color='green', label='Predict')
    plt.xlabel('the number of test data')
    plt.ylabel('earn_rate')
    plt.title('2020.1—2021.5')
    plt.legend()
    plt.show()
