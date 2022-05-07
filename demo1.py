from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    data = load_iris()
    print("莺尾花数据集：\n", data)
    print("莺尾花数据集介绍：\n", data.DESCR)
    print("莺尾花数据集特征：\n", data.feature_names)
    print("莺尾花数据集特征：\n", data.feature_names)
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)
    print(x_train)
    return None


def datasets_demo1():
    """
    数据集划分
    :return:
    """

    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)
    print(x_train)
    print(type(x_train))  # <class 'numpy.ndarray'>


def datasets_demo2():
    """
    特征工程，将数据进行特征提取
    字典
    :return:
    """

    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名称:\n", transfer.get_feature_names())
    return None


def datasets_demo3():
    """

    :return:
    """

    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名称:\n", transfer.get_feature_names())
    return None


if __name__ == '__main__':
    datasets_demo2()
