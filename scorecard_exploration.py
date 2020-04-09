# -*- coding: utf-8 -*-
"""
@ Author: xubingfeng
@ Date: 2020-03-11 22:57:24
@ Description: 数据探查模块
"""

import os
import sys
import json
import getopt
import warnings
import collections
import numpy as np
import pandas as pd
from seaborn.distributions import _statsmodels_univariate_kde, _scipy_univariate_kde, _freedman_diaconis_bins

try:
    import statsmodels.nonparametric.api as smnp

    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False


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


def univariate_kdeplot(data, kernel="gau", bw="scott", gridsize=100, cut=3, clip=None, cumulative=False):
    # Sort out the clipping
    if clip is None:
        clip = (-np.inf, np.inf)

    # Calculate the KDE
    if _has_statsmodels:
        # Prefer using statsmodels for kernel flexibility
        x, y = _statsmodels_univariate_kde(data, kernel, bw,
                                           gridsize, cut, clip,
                                           cumulative=cumulative)
    else:
        # Fall back to scipy if missing statsmodels
        if kernel != "gau":
            kernel = "gau"
            msg = "Kernel other than `gau` requires statsmodels."
            warnings.warn(msg, UserWarning)
        if cumulative:
            raise ImportError("Cumulative distributions are currently"
                              "only implemented in statsmodels."
                              "Please install statsmodels.")
        x, y = _scipy_univariate_kde(data, bw, gridsize, cut, clip)

    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)
    return x, y


def find_bin(bins, value):
    if value == bins[-1]:
        return bins[-2]
    l, r = 0, len(bins)
    while l < r:
        mid = (l + r) // 2
        if value >= bins[mid]:
            l = mid + 1
        else:
            r = mid
    return bins[l - 1]


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
    if feature_value_length < threshold_count:
        return 'CATEGORY'
    if feature_value_length / (real_data_length + 1) < threshold_rate:
        return 'CATEGORY'
    return 'NUMERIC'


def representation_checkout(df):
    """
    @ description: 探查dataframe数据集中每列的数值表现类型，是数值型还是离散型
    @ param df {dataframe} 输入的数据集
    @ return: {dataframe} 数据集特征的数值表现
    """
    threshold_rate = 0.00001
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


def describe(data, percentiles=None, include='number'):
    percentiles = np.array([0.25, 0.5, 0.75])

    # sort and check for duplicates
    unique_pcts = np.unique(percentiles)
    if len(unique_pcts) < len(percentiles):
        raise ValueError("percentiles cannot contain duplicates")
    percentiles = unique_pcts

    from pandas.io.formats.format import format_percentiles
    formatted_percentiles = format_percentiles(percentiles)

    def describe_numeric_1d(series):
        stat_index = (['count', 'mean', 'std', 'min'] +
                      formatted_percentiles + ['max'])
        d = ([series.count(), series.mean(), series.std(), series.min()] +
             series.quantile(percentiles).tolist() + [series.max()])
        return pd.Series(d, index=stat_index, name=series.name)

    def describe_categorical_1d(data):
        names = ['count', 'unique']
        objcounts = data.value_counts()
        count_unique = len(objcounts[objcounts != 0])
        result = [data.count(), count_unique]
        if result[1] > 0:
            top, freq = objcounts.index[0], objcounts.iloc[0]
            names += ['top', 'freq']
            result += [top, freq]

        return pd.Series(result, index=names, name=data.name)

    rep_df = representation_checkout(data)
    numeric_variable = rep_df[rep_df['representation'] == 'NUMERIC']['variable']
    category_variable = rep_df[rep_df['representation'] == 'CATEGORY']['variable']
    numeric_data = data.loc[:, list(numeric_variable)]
    category_data = data.loc[:, list(category_variable)]

    # when some numerics are found, keep only numerics
    data0 = numeric_data if include == 'number' else category_data
    if len(data.columns) == 0:
        data0 = data

    data = data0
    ldesc = [describe_numeric_1d(s) if include == 'number' else describe_categorical_1d(s) for _, s in data.iteritems()]
    # set a convenient order for rows
    names = []
    ldesc_indexes = sorted((x.index for x in ldesc), key=len)
    for idxnames in ldesc_indexes:
        for name in idxnames:
            if name not in names:
                names.append(name)

    d = pd.concat(ldesc, join_axes=pd.Index([names]), axis=1)
    d.columns = data.columns.copy()
    return d


def dataset_head(hr):
    return hr.head()


def dataset_shape(hr, file_path, hr_flag):
    shape = hr.shape
    dataset_name = file_path.split('/')[-1].split('.')[0]
    label_number = hr.columns.values.tolist().index(hr_flag)
    info_list = []
    hr_flag_value_counts = hr[hr_flag].value_counts().to_dict()
    hr_flag_sum = sum(list(hr_flag_value_counts.values()))
    for value, count in hr_flag_value_counts.items():
        info_dict = dict()
        info_dict['labelSort'] = value
        info_dict['labelSortCount'] = hr_flag_sum
        info_dict['labelSortValue'] = count
        info_dict['labelSortRate'] = count / shape[0]
        info_list.append(info_dict)

    ret_df = pd.DataFrame(
        {'datasetShape': [dataset_name, shape[0], shape[1], hr_flag, label_number, info_list]},
        index=['dataSetName', 'dataSetSampleCount', 'dataSetFeaturesCount', 'dataSetLabelFeaturesName',
               'dataSetLabelFeaturesNum', 'dataSetLabelInfo']
    )

    return ret_df


def data_details(hr, hr_flag):
    hr_dd = pd.DataFrame([hr.dtypes[i].name for i in range(len(hr.dtypes))], columns=['featureNumpyDtype'],
                         index=list(hr.dtypes.index))
    hr_dd['featureNunique'] = hr.nunique()
    hr_dd['featureMissingValues'] = hr.isnull().sum()
    hr_dd['featureLossRate'] = hr_dd['featureMissingValues'] / hr.shape[0]
    hr_dd['featureZeroValues'] = (hr == 0).sum()
    hr_dd['featureMode'] = hr.mode().loc[0, :]
    hr_dd = hr_dd.drop(hr_flag, axis=0)
    return hr_dd


def data_describe(hr, hr_flag):
    # feature_type_unique = hr.dtypes.unique()
    ret_dict = {}

    # if np.dtype('int') in feature_type_unique or np.dtype('float') in feature_type_unique:
    describe_numeric = describe(hr, include='number')

    describe_numeric.loc['featureVar', :] = describe_numeric.loc['std', :] ** 2

    top3_list = [hr[col].value_counts(normalize=True).to_dict() for col in describe_numeric.columns.values]
    top3_rate_list = []
    for tl in top3_list:
        temp_list = []
        value_list = list(tl.keys())
        if len(value_list) == 1:
            temp_list.append({'key': value_list[0], 'value': 1.00})
        elif len(value_list) == 2:
            temp_list.append({'key': value_list[0], 'value': tl[value_list[0]]})
            temp_list.append({'key': value_list[1], 'value': tl[value_list[1]]})
        else:
            for value in value_list[: 2]:
                temp_list.append({'key': value, 'value': tl[value]})
            temp_list.append({'key': '其他', 'value': 1 - sum([tl[value] for value in value_list[: 2]])})
        top3_rate_list.append(temp_list)
    top3_numeric = pd.DataFrame(dict(zip(describe_numeric.columns.values, [[trl] for trl in top3_rate_list])),
                                index=['top3'])

    range_numeric = pd.DataFrame(dict(zip(describe_numeric.columns.values,
                                          ['[' + str(mi) + ', ' + str(ma) + ']' for mi, ma in list(
                                              zip(describe_numeric.loc['min', :].tolist(),
                                                  describe_numeric.loc['max', :].tolist()))])),
                                 index=['featureRange'])

    value_counts_numeric = pd.DataFrame(dict(zip(describe_numeric.columns.values,
                                                 [[hr[col].value_counts().to_dict()]
                                                  for col in describe_numeric.columns.values])),
                                        index=['featureValueCounts'])

    name_numeric = pd.DataFrame(dict(zip(describe_numeric.columns.values, describe_numeric.columns.values.tolist())),
                                index=['featureName'])

    describe_numeric = describe_numeric.append(name_numeric)
    describe_numeric = describe_numeric.append(top3_numeric)
    describe_numeric = describe_numeric.append(range_numeric)
    describe_numeric = describe_numeric.append(value_counts_numeric)

    describe_numeric.rename({'count': "featureCount",
                             'mean': "featureMean",
                             'std': "featureStd",
                             'min': "featureMin",
                             '25%': "featurePer25",
                             '50%': "featurePer50",
                             '75%': "featurePer75",
                             'max': "featureMax",
                             },
                            inplace=True)

    distribution_list = []
    for feature in describe_numeric.columns.values:

        if isinstance(hr[feature], list):
            hr[feature] = np.asarray(hr[feature])
        hr[feature] = hr[feature].astype(np.float64)
        x, y = univariate_kdeplot(hr[feature])
        kde_list = list(zip(x.tolist(), y.tolist()))

        bins = min(_freedman_diaconis_bins(hr[feature]), 50)
        m, bins = np.histogram(hr[feature], bins=bins, density=True)
        m, bins = m.tolist(), bins.tolist()
        devided_number = (bins[1] - bins[0]) * len(hr[feature])

        temp_list = []
        distribution_dict = dict()

        feature_value_counts_dict = hr[feature].value_counts().to_dict()
        feature_value_list = list(feature_value_counts_dict.keys())

        positive_list, negative_list = [], []

        positive_dict = collections.OrderedDict()
        negative_dict = collections.OrderedDict()
        for bi in bins[:-1]:
            positive_dict[bi] = 0
        for bi in bins[:-1]:
            negative_dict[bi] = 0

        for feature_value in feature_value_list:
            pos_neg_value_counts = hr[hr[feature] == feature_value][hr_flag].value_counts().to_dict()
            value_bin = find_bin(bins, feature_value)
            if 0 in pos_neg_value_counts.keys():
                negative_dict[value_bin] += pos_neg_value_counts[0]
            if 1 in pos_neg_value_counts.keys():
                positive_dict[value_bin] += pos_neg_value_counts[1]

        for k in positive_dict.keys():
            positive_dict[k] /= devided_number
        for k in negative_dict.keys():
            negative_dict[k] /= devided_number

        for k, v in positive_dict.items():
            positive_list.append((k, v))
        for k, v in negative_dict.items():
            negative_list.append((k, v))

        distribution_dict['feature_name'] = feature
        distribution_dict['feature_details'] = {'positive': positive_list, 'negative': negative_list, 'kde': kde_list}
        temp_list.append(distribution_dict)
        distribution_list.append(temp_list)

    numeric_distributions = pd.DataFrame(dict(zip(describe_numeric.columns.values, distribution_list)),
                                         index=['featureFreqs'])

    describe_numeric = describe_numeric.append(numeric_distributions)
    ret_dict['describe_numeric'] = describe_numeric

    # if np.dtype('O') in feature_type_unique:
    describe_category = describe(hr, include='object')

    top3_list = [hr[col].value_counts(normalize=True).to_dict() for col in describe_category.columns.values]
    top3_rate_list = []
    for tl in top3_list:
        temp_list = []
        value_list = list(tl.keys())
        if len(value_list) == 1:
            temp_list.append({'key': value_list[0], 'value': 1.00})
        elif len(value_list) == 2:
            temp_list.append({'key': value_list[0], 'value': tl[value_list[0]]})
            temp_list.append({'key': value_list[1], 'value': tl[value_list[1]]})
        else:
            for value in value_list[: 2]:
                temp_list.append({'key': value, 'value': tl[value]})
            temp_list.append({'key': '其他', 'value': 1 - sum([tl[value] for value in value_list[: 2]])})
        top3_rate_list.append(temp_list)
    top3_category = pd.DataFrame(dict(zip(describe_category.columns.values, [[trl] for trl in top3_rate_list])),
                                 index=['top3'])

    col_values = [list(hr[col].value_counts().index) for col in describe_category.columns.values]
    col_values_modified = []
    for col_value in col_values:
        col_values_modified.append(map(str, col_value))
    col_values_modified = [', '.join(cvm) for cvm in col_values_modified]

    range_category = pd.DataFrame(dict(zip(describe_category.columns.values, col_values_modified)),
                                  index=['featureRange'])

    value_counts_category = pd.DataFrame(dict(zip(describe_category.columns.values,
                                                  [[hr[col].value_counts().to_dict()]
                                                   for col in describe_category.columns.values])),
                                         index=['featureValueCounts'])

    name_category = pd.DataFrame(dict(zip(describe_category.columns.values, describe_category.columns.values.tolist())),
                                 index=['featureName'])

    describe_category = describe_category.append(name_category)
    describe_category = describe_category.append(top3_category)
    describe_category = describe_category.append(range_category)
    describe_category = describe_category.append(value_counts_category)

    describe_category.rename({'count': "featureCount",
                              'unique': "featureUnique",
                              'top': "featureTop",
                              'freq': "featureFreq"
                              },
                             inplace=True)

    distribution_list = []
    for feature in describe_category.columns.values:
        temp_list = []
        distribution_dict = dict()
        feature_value_counts_dict = hr[feature].value_counts().to_dict()
        feature_value_list = list(feature_value_counts_dict.keys())
        positive_list, negative_list = [], []
        for feature_value in feature_value_list:
            pos_neg_value_counts = hr[hr[feature] == feature_value][hr_flag].value_counts().to_dict()
            if 0 in pos_neg_value_counts.keys():
                negative_list.append((feature_value, pos_neg_value_counts[0]))
            if 1 in pos_neg_value_counts.keys():
                positive_list.append((feature_value, pos_neg_value_counts[1]))
        distribution_dict['feature_name'] = feature
        distribution_dict['feature_details'] = {'positive': positive_list, 'negative': negative_list}
        temp_list.append(distribution_dict)
        distribution_list.append(temp_list)

    category_distributions = pd.DataFrame(dict(zip(describe_category.columns.values, distribution_list)),
                                          index=['featureFreqs'])

    describe_category = describe_category.append(category_distributions)
    ret_dict['describe_category'] = describe_category

    return ret_dict


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def data_exploration(file_path, output_path, flag, file_size=100):
    hr = pd.read_csv(file_path)
    dataset_name = file_path.split('/')[-1].split('.')[0]
    mkdir(os.path.join(output_path, dataset_name))
    hr_shape = dataset_shape(hr, file_path, flag)
    hr_dd = data_details(hr, flag).stack().unstack(0)
    hr_des = data_describe(hr, flag)

    with open(os.path.join(output_path, dataset_name, 'dataset_details.json'), 'w', encoding='utf-8') as f:
        json.dump(list(hr_shape.to_dict(orient='dict').values())[0], f, cls=NumpyEncoder, ensure_ascii=False)

    if 'describe_numeric' in hr_des.keys():
        describe_numeric = pd.concat([hr_dd, hr_des['describe_numeric']], sort=False, join='inner')
        length = len(describe_numeric.columns.values)
        i = 0
        while length >= file_size:
            with open(os.path.join(output_path, dataset_name, 'feature_continuance_{}.json'.format(i)), 'w',
                      encoding='utf-8') as f:
                json.dump(list(describe_numeric.iloc[:, i * file_size: (i + 1) * file_size - 1].to_dict(orient='dict').
                               values()), f, cls=NumpyEncoder, ensure_ascii=False)
            length -= file_size
            i += 1
        if length == 0:
            pass
        else:
            with open(os.path.join(output_path, dataset_name, 'feature_continuance_{}.json'.format(i)), 'w',
                      encoding='utf-8') as f:
                json.dump(list(describe_numeric.iloc[:, i * file_size:].to_dict(orient='dict').values()), f,
                          cls=NumpyEncoder, ensure_ascii=False)

    if 'describe_category' in hr_des.keys():
        describe_category = pd.concat([hr_dd, hr_des['describe_category']], sort=False, join='inner')
        length = len(describe_category.columns.values)
        i = 0
        while length >= file_size:
            with open(os.path.join(output_path, dataset_name, 'feature_discrete_{}.json'.format(i)), 'w',
                      encoding='utf-8') as f:
                json.dump(list(describe_category.iloc[:, i * file_size: (i + 1) * file_size - 1].to_dict(orient='dict').
                               values()), f, cls=NumpyEncoder, ensure_ascii=False)
            length -= file_size
            i += 1
        if length == 0:
            pass
        else:
            with open(os.path.join(output_path, dataset_name, 'feature_discrete_{}.json'.format(i)), 'w',
                      encoding='utf-8') as f:
                json.dump(list(describe_category.iloc[:, i * file_size:].to_dict(orient='dict').values()), f,
                          cls=NumpyEncoder, ensure_ascii=False)


if __name__ == "__main__":
    # 注意路径使用'/'分割
    opts, args = getopt.getopt(sys.argv[1:], "i:o:l:")
    dataset_path = opts[0][1]
    output_path = opts[1][1]
    label = opts[2][1]
    import time
    s = time.time()
    data_exploration(dataset_path, output_path, label)
    e = time.time()
    print(e - s)

    # import time
    #
    # file_list = os.listdir(r'E:\workspace\jupyter\YCKJ\data\data')
    # time_list = []
    # for i in range(len(file_list)):
    #     print(file_list[i])
    #     s = time.time()
    #     data_exploration(r'E:\workspace\jupyter\YCKJ\data\data' + "\\" + file_list[i],
    #                      r"E:\workspace\jupyter\YCKJ\data\data", "bad")
    #     e = time.time()
    #     time_list.append(e - s)
    #     print(e-s)
    #     print("-----------------------------------------------------")
    # print(time_list)

