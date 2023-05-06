import pandas as pd
from pandas import read_excel
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
import joblib

# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# 定义字符串转换为浮点型（此处是转换换手率）
def str_to_float(s):
    s = s[:-1]
    s_float = float(s)
    return s_float


def data_read(path):
    """
    处理读取数据，构建收益率特征列
    :param path:
    :return:
    """
    # 读取excel文件，并将‘日期’列解析为日期时间格式,并设为索引
    print("*" * 20)
    print('读取：', path)
    data = read_excel(path, parse_dates=['日期'], index_col='日期')

    # 对数据的列名重新命名
    data.columns = ['close']
    data.index.name = 'date'  # 日期为索引列
    # 将数据按日期这一列排序（保证后续计算收益率的正确性）
    data = data.sort_values(by='date')
    # 增加一列'earn_rate', 存储每日的收益率
    data['earn_rate'] = data['close'].pct_change()
    # 缺失值填充
    data['earn_rate'].fillna(method='bfill', inplace=True)
    print(data.head())
    print("数据量：", len(data))
    return data


def data_concat(data_1, data_2, data_3):
    """
    合并3个数据,将3种类型大宗商品数据合并，互相预测
    :param data_1:
    :param data_2:
    :param data_3:
    :return:
    """
    # 合并数据
    data_ls = pd.concat([data_1, data_2], axis=1)
    data = pd.concat([data_ls, data_3], axis=1)
    # 重命名列
    data.columns = ['gjs_close', 'gjs_earn', 'hg_close', 'hg_earn', 'ys_close', 'ys_earn']
    print("*" * 20)
    print(data.head())
    print("数据量：", len(data))
    return data


def min_max(data):
    """
    数据归一化
    :param data:
    :return:
    """
    # 获取DataFrame中的数据，形式为数组array形式
    values = data.values
    # 确保所有数据为float类型
    values = values.astype('float32')

    # 特征的归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # 保存归一化模型
    model_path = 'models/MinMaxScaler_model.pkl'
    joblib.dump(scaler, model_path)
    print("*" * 20)
    print("完成归一化处理：")
    print(scaled[:5])
    print("完成归一化模型保存：", model_path)
    return scaled


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    将时间序列转换为监督学习问题
    Arguments:
        data: 输入数据需要是列表或二维的NumPy数组的观察序列。
        n_in: 输入的滞后观察数（X）。值可以在[1..len（data）]之间，可选的。默认为1。
        n_out: 输出的观察数（y）。值可以在[0..len（data）-1]之间，可选的。默认为1。
        dropnan: Bool值，是否删除具有NaN值的行，可选的。默认为True。
    Returns:
        用于监督学习的Pandas DataFrame。
    """
    # 定义series_to_supervised()函数
    # 将时间序列转换为监督学习问题
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # 删除多余列
    agg.drop(agg.columns[[6, 8, 10]], axis=1, inplace=True)
    print("*" * 20)
    print("完成监督学习转换：")
    print(agg.head())
    return agg


if __name__ == '__main__':
    # 读取贵金属
    data_path = 'data/data_贵金属.xlsx'
    data_gjs = data_read(data_path)
    # 读取化工
    data_path = 'data/data_化工.xlsx'
    data_hg = data_read(data_path)
    # 读取有色
    data_path = 'data/data_有色.xlsx'
    data_ys = data_read(data_path)
    # 合并数据
    data_c = data_concat(data_gjs, data_hg, data_ys)
    # 数据归一化
    data_mx = min_max(data_c)
    # 将时间序列转换为监督学习问题
    re_framed = series_to_supervised(data_mx, 1, 1)
    data_save_path = "data/pre_data.csv"
    re_framed.to_csv(data_save_path, index=False)
    print("完成数据预处理，保存地址：", data_save_path)
