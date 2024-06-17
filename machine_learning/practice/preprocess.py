"""
Author: jhzhu
Date: 2024/6/16
Description: 
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

ENCODER_COLUMNS = ['租房网站名称', '小区', '城市', '区', '朝向']
# FILL_VALUE = ['南,东西,西南,东,西,东北,西北,北']

def custom_adjusted_r2(y_true, y_pred, **kwargs):
    if 'x_column' not in kwargs['kwargs']:
        return 0
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    return 1 - (1 - r2) * (len(y_pred) - 1) / (len(y_pred) - kwargs['kwargs']['x_column'] - 1)


def custom_error_percentage(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        return round(1 - np.sum(np.abs(y_true.values.flatten() - y_pred)) / np.sum(y_true), 3)[0]
    else:
        return round(1 - np.sum(np.abs(y_true - y_pred)) / np.sum(y_true), 3)


def custom_error_percentage_avg(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    return np.round(np.mean(1 - np.abs(y_true - y_pred) / y_true), 3)


def evaluate_predict_result(x, y_true, y_pred):
    result_dict = dict()
    result_dict['mean_absolute_error'] = round(mean_absolute_error(y_true=y_true, y_pred=y_pred), 3)
    result_dict['median_absolute_error'] = round(median_absolute_error(y_true=y_true, y_pred=y_pred), 3)
    result_dict['root_mean_squared_error'] = round(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 3)
    result_dict['r2'] = round(r2_score(y_true=y_true, y_pred=y_pred), 3)
    result_dict['adjusted_r2'] = round(custom_adjusted_r2(y_true, y_pred, kwargs={'x_column': x.shape[1]}), 3)
    result_dict['accuracy'] = round(custom_error_percentage(y_true, y_pred), 3)
    result_dict['avg accuracy'] = round(custom_error_percentage_avg(y_true, y_pred), 3)
    return result_dict


def plot_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns

    indices = np.argsort(importances)[::-1]
    plt.rcParams['font.sans-serif'] = ['Yuanti SC']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=45)

    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.grid()
    plt.show()


def zero_index_features(df, columns_to_reindex):
    for column in columns_to_reindex:
        df[column] = pd.factorize(df[column])[0]


def get_column_transformer(encoded_columns_name):
    return ColumnTransformer([('encoder', OneHotEncoder(drop='first'), encoded_columns_name)], remainder='passthrough')


rent_house_df = pd.read_csv('../../dataset/中国租房信息数据集.csv')
filtered_df = rent_house_df.drop(columns=['link', '详细地址', 'lng', 'lat', '信息发布人类型']).dropna()
filtered_df = filtered_df[(rent_house_df['面积'] >= 5) & (filtered_df['面积'] / filtered_df['室'] >= 3)]

zero_index_features(filtered_df, ENCODER_COLUMNS)
X_train, X_test, y_train, y_test = train_test_split(filtered_df.loc[:, filtered_df.columns != '价格'],
                                                    filtered_df.loc[:, '价格'],
                                                    test_size=0.3, random_state=42, shuffle=True)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
plot_feature_importance(rf, X_train)

rf_pred = rf.predict(X_test)
result = evaluate_predict_result(X_test, y_test, rf_pred)
print(result)
