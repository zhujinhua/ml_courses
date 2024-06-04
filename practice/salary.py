import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

salary_df = pd.read_csv('成人收入预测数据集.csv')
use_columns = [col for col in salary_df.columns if col != 'fnlwgt']
# skip fnlwgt
logging.info(salary_df[:, use_columns])
salary_df.iloc[:,]