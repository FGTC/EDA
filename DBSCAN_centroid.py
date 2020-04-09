import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def data_distribution_type(hr, hr_flag):
    rep_df = representation_checkout(hr)
    numeric_columns = list(rep_df[rep_df['representation'] == 'NUMERIC']['variable'])
    hr = hr[numeric_columns+[hr_flag]].reset_index(drop=True)
    numeric_count = hr.shape[1]
    hr_flagcolnum = list(hr.columns).index(hr_flag)
    result_0 = hr[hr[hr_flag].isin(['0'])].drop([hr_flag], axis=1)
    result_1 = hr[hr[hr_flag].isin(['1'])].drop([hr_flag], axis=1)
    result_count = hr[hr_flag].value_counts().to_dict()
    result_count_0 = int(result_count[0.0])
    result_count_1 = int(result_count[1.0])
    if len(result_0) > 200:
        result_count_0 = int((len(result_0) - 200) * 0.2 + 200)
        if result_count_0 > 400:
            result_count_0 = 400
    if len(result_1) > 200:
        result_count_1 = int((len(result_1) - 200) * 0.2 + 200)
        if result_count_1 > 400:
            result_count_1 = 400
    result_list = []
    if len(result_0) > 200 and len(result_1) > 200:
        rr_0 = DBSCAN_centroid(result_0, result_count_0, 0, hr_flagcolnum)
        rr_1 = DBSCAN_centroid(result_1, result_count_1, 1, hr_flagcolnum)
        rr_0.extend(rr_1)
        result = np.array(rr_0)
        for I in range(numeric_count):
            if I == hr_flagcolnum:
                continue
            for i in range(I + 1, numeric_count):
                if i == hr_flagcolnum:
                    continue
                dict1 = {"feature_name_1": hr.columns.values[I], "feature_name_2": hr.columns.values[i],"label_name": hr.columns.values[hr_flagcolnum]}
                dict1["feature_details"] = result[:, [I, i]]
                result_list.append(dict1)
    elif len(result_0) <= 200 and len(result_1) > 200:
        rr_1 = DBSCAN_centroid(result_1, result_count_1, 1, hr_flagcolnum)
        for I in range(numeric_count):
            if I == hr_flagcolnum:
                continue
            for i in range(I + 1, numeric_count):
                if i == hr_flagcolnum:
                    continue
                rr_0 = np.array(result_0.loc[:, [hr.columns.values[I], hr.columns.values[i], hr_flag]])
                rr_11 = np.array(rr_1)[:, [I, i, hr_flagcolnum]]
                result = np.insert(rr_11, 0, values=rr_0, axis=0).tolist()
                dict1 = {"feature_name_1": hr.columns.values[I], "feature_name_2": hr.columns.values[i],"label_name": hr.columns.values[hr_flagcolnum]}
                dict1["feature_details"] = result
                result_list.append(dict1)
    elif len(result_0) > 200 and len(result_1) <= 200:
        rr_0 = DBSCAN_centroid(result_0, result_count_0, 0, hr_flagcolnum)
        for I in range(numeric_count):
            if I == hr_flagcolnum:
                continue
            for i in range(I + 1, numeric_count):
                if i == hr_flagcolnum:
                    continue
                rr_00 = np.array(rr_0)[:, [I, i, hr_flagcolnum]]
                rr_1 = np.array(result_1.loc[:, [hr.columns.values[I], hr.columns.values[i], hr_flag]])
                result = np.insert(rr_1, 0, values=rr_00, axis=0).tolist()
                dict1 = {"feature_name_1": hr.columns.values[I], "feature_name_2": hr.columns.values[i],"label_name": hr.columns.values[hr_flagcolnum]}
                dict1["feature_details"] = result
                result_list.append(dict1)
    elif len(result_0) <= 200 and len(result_1) <= 200:
        for I in range(numeric_count):
            if I == hr_flagcolnum:
                continue
            for i in range(I + 1, numeric_count):
                if i == hr_flagcolnum:
                    continue
                rr_0 = np.array(result_0.loc[:, [hr.columns.values[I], hr.columns.values[i], hr_flag]])
                rr_1 = np.array(result_1.loc[:, [hr.columns.values[I], hr.columns.values[i], hr_flag]])
                result = np.insert(rr_1, 0, values=rr_0, axis=0).tolist()
                dict1 = {"feature_name_1": hr.columns.values[I], "feature_name_2": hr.columns.values[i],"label_name": hr.columns.values[hr_flagcolnum]}
                dict1["feature_details"] = result
                result_list.append(dict1)
    return result_list


def DBSCAN_centroid(hr, num_clusters, trget, hr_flagcolnum):
    X = hr
    X = StandardScaler().fit_transform(X) # StandardScaler作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
    # #############################################################################
    # 调用密度聚类  DBSCAN
    db = DBSCAN(eps=0.2, min_samples=10).fit(X)
    # print(db.labels_)  # db.labels_为所有样本的聚类索引，没有聚类索引为-1
    # print(db.core_sample_indices_) # 所有核心样本的索引
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # 设置一个样本个数长度的全false向量
    core_samples_mask[db.core_sample_indices_] = True #将核心样本部分设置为true
    labels = db.labels_
    print(len(set(labels)))
    # 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_percent = n_clusters_/len(hr)
    # #############################################################################
    # Plot result
    # 使用黑色标注离散点
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    XY_1 = []
    XY_0 = []
    for k, col in zip(unique_labels, colors):
        if k == -1:  # 聚类结果为-1的样本为离散点
            # 使用黑色绘制离散点
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)  # 将所有属于该聚类的样本位置置为true

        xy_1 = X[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
        xy_0 = X[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
        xy_0 = np.insert(xy_0, hr_flagcolnum, values=trget, axis=1)
        xy_1 = np.insert(xy_1, hr_flagcolnum, values=trget, axis=1)
        if len(xy_1)*n_percent<=1:  # 如果按比例取值小于1,取蔟均值
            xy_1 = np.mean(xy_1, axis=0).tolist()
        else:  # 如果按比例取值大于1,蔟间隔采样
            num = len(xy_1)*n_percent
            xy_1 = np.linspace(xy_1[1], xy_1[-1], num=int(num)).tolist()
        if len(xy_0)*n_percent<=1:  # 如果按比例取值小于1,取蔟均值
            xy_0 = np.mean(xy_0, axis=0).tolist()
        else:  # 如果按比例取值大于1,蔟间隔采样
            num = len(xy_0)*n_percent
            xy_0 = np.linspace(xy_0[1], xy_0[-1], num=int(num)).tolist()
        # plt.plot(xy_1[:, 0], xy_1[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        # plt.plot(xy_0[:, 0], xy_0[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        XY_1.append(xy_1)
        XY_0.append(xy_0)
    XY_1.append(XY_0)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    XY_1 = list(dfs(XY_1))
    return XY_1


def dfs(tree):
    """
    :param tree: 输入多维array
    :return: 返回二维array
    """
    for i in tree:
        if type(i) == list:
            yield from dfs(i)
        else:
            yield tree


def _verify(sg_col_df, threshold_rate, threshold_count):
    """
    @ description: 数值、离散的划分规则
    @ param sg_col_df {dataframe} 单列的dataframe，用于检查该列的数值表现
    @ param threshold_rate {float} 占比阈值，如果特征值域元素的个数在总样本数的占比小于占比阈值，那么判定该特征为离散型
    @ param threshold_count {int} 个数阈值，如果特征值域元素的个数小于个数阈值，那么判定该特征为离散型
    @ return: {str} 返回数值离散的标志字符串：'CATEGORY' or 'NUMERIC'
    """
    real_data = sg_col_df.dropna()
    real_data_length = real_data.shape[0]
    feature_value = real_data.unique()
    feature_value_length = len(feature_value)
    if feature_value_length / (real_data_length + 1) < threshold_rate:
        return 'CATEGORY'
    if feature_value_length < threshold_count:
        return 'CATEGORY'
    return 'NUMERIC'


def representation_checkout(df):
    """
    @ description: 探查dataframe数据集中每列的数值表现类型，是数值型还是离散型
    @ param df {dataframe} 输入的数据集
    @ return: {dataframe} 数据集特征的数值表现
    """
    threshold_rate = 0.001
    threshold_count = 20
    variable = []
    feature_type = []

    for col in df.columns:
        flag = "CATEGORY"  # NUMERIC,CATEGORY 默认为类别特征
        if df[col].dtype == np.object:
            try:
                df[col] = df[col].astype(np.int)
                flag = _verify(df[col], threshold_rate, threshold_count)
            except ValueError:
                try:
                    df[col] = df[col].astype(np.float)
                    flag = _verify(df[col], threshold_rate, threshold_count)
                except ValueError:
                    pass
        elif df[col].dtype == np.int64:
            flag = _verify(df[col], threshold_rate, threshold_count)
        elif df[col].dtype == np.float:
            flag = _verify(df[col], threshold_rate, threshold_count)
        else:
            # 包括bool、String、unicode等，目前均归属于类别特征
            pass
        variable.append(col)
        feature_type.append('NUMERIC' if flag == "NUMERIC" else 'CATEGORY')

    rep_df = pd.DataFrame({
        'variable': variable,
        'representation': feature_type
    })

    return rep_df


def data_exploration(file_path, output_path, flag):
    hr = pd.read_csv(file_path)
    dataset_name = file_path.split('/')[-1].split('.')[0]
    filedir = os.path.join(output_path, dataset_name, 'feature_comb.json')
    hr_distribution_type = data_distribution_type(hr, flag)
    with open(filedir, "w", encoding='utf-8') as f:
        json.dump(hr_distribution_type, f, cls=NumpyEncoder, ensure_ascii=False)


if __name__ == "__main__":
    dataset_path = 'D:\实习\云从科技\EDA\data\head_500.csv'
    output_path = 'D:\实习\云从科技\EDA\data'
    data_exploration(dataset_path, output_path, 'bad')
    # 注意路径使用'/'分割
    # opts, args = getopt.getopt(sys.argv[1:], "i:o:l:")
    # dataset_path = opts[0][1]
    # output_path = opts[1][1]
    # label = opts[2][1]
    # data_exploration(dataset_path, output_path, label)