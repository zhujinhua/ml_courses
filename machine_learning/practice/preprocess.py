"""
Author: jhzhu
Date: 2024/6/16
Description: 
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


ENCODER_COLUMNS = ['租房网站名称', '小区', '城市', '区', '朝向']
FILL_VALUE = ['南’, ‘东西’, ‘西南’, ‘东’, ‘西’, ‘东北’, ‘西北’, ‘北']


def fill_na_with_list(series, values):
    na_indices = series[series.isna()].index
    random_values = np.random.choice(values, size=len(na_indices))
    series.loc[na_indices] = random_values
    return series


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
    return ColumnTransformer([('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), encoded_columns_name)], remainder='passthrough')


rent_house_df = pd.read_csv('../../dataset/中国租房信息数据集.csv')
# rent_house_df['朝向'] = fill_na_with_list(rent_house_df['朝向'], FILL_VALUE)
filtered_df = rent_house_df.drop(columns=['link', '详细地址', '信息发布人类型']).dropna()
# filtered_df = filtered_df[(rent_house_df['面积'] >= 5) & (filtered_df['面积'] / filtered_df['室'] >= 3)]
# filtered_df.loc[filtered_df['周边学校个数'] >= 30, '周边学校个数'] = 30
# filtered_df.loc[filtered_df['周边医院个数'] >= 30, '周边医院个数'] = 30

zero_index_features(filtered_df, ENCODER_COLUMNS)
X_train, X_test, y_train, y_test = train_test_split(filtered_df.loc[:, filtered_df.columns != '价格'],
                                                    filtered_df.loc[:, '价格'],
                                                    test_size=0.3, random_state=42, shuffle=True)
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
plot_feature_importance(rf, X_train)

rf_pred = rf.predict(X_test)
result = evaluate_predict_result(X_test, y_test, rf_pred)
print(result)

# estimator = []
# # estimator.append(('SVM', SVR()))
# estimator.append(('Linear Regressor', LinearRegression()))
# # estimator.append(('RF', RandomForestRegressor(random_state=42)))
# voting = VotingRegressor(estimators=estimator)
#
# onehot = OneHotEncoder(handle_unknown='ignore')
# X = filtered_df.loc[:, filtered_df.columns != '价格']
# onehot.fit(X[ENCODER_COLUMNS])
# X_encoded = onehot.transform(X[ENCODER_COLUMNS])
#
# X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=onehot.get_feature_names_out(ENCODER_COLUMNS))
# X_remaining = X.drop(columns=ENCODER_COLUMNS).reset_index(drop=True)
# X_full = pd.concat([X_encoded_df, X_remaining], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X_full,
#                                                     filtered_df.loc[:, '价格'],
#                                                     test_size=0.3, random_state=42, shuffle=True)
# pipeline = Pipeline([
#     ('Standardization', StandardScaler(with_mean=False)),
#     # ('SVD', TruncatedSVD(n_components=90)),
#     ('Estimator', voting)
# ])
# pipeline.fit(X_train, y_train)
# voting_pred = pipeline.predict(X_test)
# result = evaluate_predict_result(X_test, y_test, voting_pred)
# print(result)
