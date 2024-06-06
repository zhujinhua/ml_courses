import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_transformer(encoded_columns_name):
    return ColumnTransformer([('encoder', OneHotEncoder(drop='first'), encoded_columns_name)], remainder='passthrough')


logging.basicConfig(level=logging.INFO)
ENCODER_COLUMNS = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']
salary_df = pd.read_csv('成人收入预测数据集.csv')
trim_columns = [col.replace(' ', '') for col in salary_df.columns]
salary_df = salary_df.set_axis(trim_columns, axis=1)

# skip fnlwgt
use_columns = [col for col in salary_df.columns if col not in ('fnlwgt', 'salary')]
logging.info(set(salary_df.loc[:, 'salary']))
X_train, X_test, y_train, y_test = train_test_split(salary_df.loc[:, use_columns], salary_df.loc[:, 'salary'], test_size=0.2, random_state=42)
pipeline = Pipeline([
            ('Preprocess', get_column_transformer(ENCODER_COLUMNS)),
            ('Standardization', StandardScaler(with_mean=False)),
            # ('SVD', TruncatedSVD(n_components=90)),
            ('Estimator', LogisticRegression(random_state=42, max_iter=10000))
        ])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
# feature_importance = pipeline.named_steps['Estimator'].feature_importances_
# logging.info(f'Feature Importance: {feature_importance}')
logging.info(f'Accuracy: {round((y_pred == y_test).mean(), 3)}')



