import time
import os

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from pyecharts.charts import Line

# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def create_dir_not_exist(path):
    """
    文件夹创建
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def train_test(data):
    """
    构建训练集、测试集
    :param data:
    :return:
    """
    # 贵金属
    data_gjs = data.drop(['var4(t)', 'var6(t)'], axis=1)
    # 化工
    data_hg = data.drop(['var2(t)', 'var6(t)'], axis=1)
    # 有色
    data_ys = data.drop(['var2(t)', 'var4(t)'], axis=1)
    values_gjs = data_gjs.values
    values_hg = data_hg.values
    values_ys = data_ys.values
    # 构建3组训练和测试数据
    train_x_1, train_y_1, test_x_1, test_y_1 = train_create(values_gjs)
    train_x_2, train_y_2, test_x_2, test_y_2 = train_create(values_hg)
    train_x_3, train_y_3, test_x_3, test_y_3 = train_create(values_ys)
    print("完成训练数据测试数据分离......")
    return (train_x_1, train_y_1, test_x_1, test_y_1,
            train_x_2, train_y_2, test_x_2, test_y_2,
            train_x_3, train_y_3, test_x_3, test_y_3)


def train_create(values):
    """
    训练集：10年-19年
    测试集：20年-21年
    :param values:
    :return:
    """
    train = values[:2430]
    test = values[2430:]
    # 划分训练集和测试集的输入和输出
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    # 转化为三维数据
    # reshape input to be 3D [samples, timesteps, features]
    """
    Keras LSTM层的工作方式是通过接收3维（N，W，F）的数字阵列。
    其中N是训练序列的数目，W是序列长度，F是每个序列的特征数目。
    """
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


def model_create(train_X):
    """
    搭建LSTM模型
    :param train_X:
    :return:
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    return model


if __name__ == '__main__':
    data_path = 'data/pre_data.csv'
    print("从：", data_path, "。读取预处理数据......")
    data = pd.read_csv(data_path)
    print("预处理数据：")
    print(data.head())
    train_test(data)
    # 分离训练数据测试数据
    train_x_gjs, train_y_gjs, test_x_gjs, test_y_gjs, \
    train_x_hg, train_y_hg, test_x_hg, test_y_hg, train_x_ys, \
    train_y_ys, test_x_ys, test_y_ys = train_test(data)

    '''
    贵金属模型训练
    '''
    # 实例化模型
    lstm_gjs = model_create(train_x_gjs)
    # 获取系统时间
    my_time = time.strftime("%m_%d_%H_%M", time.localtime())
    # 当前路径
    work_path = os.getcwd()
    # 创建log文件夹
    my_log_dir = (work_path + "/logs/gjs")
    create_dir_not_exist(my_log_dir)

    # 定义callbacks参数
    callbacks = [
        TensorBoard(log_dir=my_log_dir)
    ]

    # 贵金属模型训练
    history1 = lstm_gjs.fit(train_x_gjs, train_y_gjs, epochs=20, batch_size=100,
                            validation_data=(test_x_gjs, test_y_gjs), callbacks=callbacks,
                            verbose=2, shuffle=False)
    # 保存最终模型
    lstm_gjs.save_weights('models/' + 'model_lstm_gjs.tf')

    '''
    化工模型训练
    '''
    # 实例化模型
    lstm_hg = model_create(train_x_hg)
    # 当前路径
    work_path = os.getcwd()
    # 创建log文件夹
    my_log_dir = (work_path + "/logs/hg")
    create_dir_not_exist(my_log_dir)

    # 定义callbacks参数
    callbacks = [
        TensorBoard(log_dir=my_log_dir)
    ]

    # 模型训练
    history2 = lstm_hg.fit(train_x_hg, train_y_hg, epochs=20, batch_size=100,
                           validation_data=(test_x_hg, test_y_hg), callbacks=callbacks,
                           verbose=2, shuffle=False)
    # 保存最终模型
    lstm_hg.save_weights('models/' + 'model_lstm_hg.tf')

    '''
    有色模型训练
    '''
    # 实例化模型
    lstm_ys = model_create(train_x_hg)
    # 当前路径
    work_path = os.getcwd()
    # 创建log文件夹
    my_log_dir = (work_path + "/logs/ys")
    create_dir_not_exist(my_log_dir)

    # 定义callbacks参数
    callbacks = [
        TensorBoard(log_dir=my_log_dir)
    ]

    # 贵金属模型训练
    history3 = lstm_ys.fit(train_x_ys, train_y_ys, epochs=20, batch_size=100,
                           validation_data=(test_x_ys, test_y_ys), callbacks=callbacks,
                           verbose=2, shuffle=False)
    # 保存最终模型
    lstm_ys.save_weights('models/' + 'model_lstm_ys.tf')

    """
        使用Tensorboard实时查看训练结果
    
        1.cd 到程序main目录放置的位置
    
        2.输入：
        tensorboard --logdir="logs/gjs"
        或
        tensorboard --logdir="logs/hg"
        或
        tensorboard --logdir="logs/ys"
    """
    # 预测结果
    y_gjs = lstm_gjs.predict(test_x_gjs)
    y_hg = lstm_hg.predict(test_x_hg)
    y_ys = lstm_ys.predict(test_x_ys)
    # 保存
    numpy.save("data/y_gjs.npy", y_gjs)
    numpy.save("data/y_hg.npy", y_hg)
    numpy.save("data/y_ys.npy", y_ys)
    numpy.save("data/test_y_gjs.npy", test_y_gjs)
    numpy.save("data/test_y_hg.npy", test_y_hg)
    numpy.save("data/test_y_ys.npy", test_y_ys)