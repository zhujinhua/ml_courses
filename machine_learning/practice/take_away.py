# 1. split the sentences
#  remove the pause words?
# 2. vector the segment
# 3. Choose training model: gaussian, train the model and get the accuracy
# 4. predict with the trained model
import logging

import jieba
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

REVIEW_DICT = {1: 'positive', 0: 'negative'}


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    return stopwords


def get_vocabulary_list(msgs, stopwords_set):
    vocab_set = set()
    for i in msgs:
        vocab_set |= set(i)
        # vocab_set |= set(i) - stopwords_set
    return sorted(list(vocab_set))


# every row is a message vector
def messages_2_vectors(vocab_list, msgs):
    msgs_len = len(msgs)
    shape = (msgs_len, len(vocab_list))
    matrix = np.zeros(shape)

    for i in range(msgs_len):
        for word in msgs[i]:
            if word in vocab_list:
                matrix[i, vocab_list.index(word)] = 1
    return matrix


logging.basicConfig(level=logging.INFO)
stopwords = load_stopwords('../../dataset/cn_stopwords.txt')
take_away_df = pd.read_csv('../../dataset/中文外卖评论数据集.csv')
take_away_df['words'] = take_away_df['review'].apply(lambda x: jieba.lcut(x.replace(' ', ''), cut_all=False))
vocabulary_set = get_vocabulary_list(take_away_df['words'], stopwords)
take_away_vecs = messages_2_vectors(vocabulary_set, take_away_df['words'])
X_train, X_test, y_train, y_test = train_test_split(take_away_vecs, take_away_df.loc[:, 'label'], test_size=0.3,
                                                    random_state=42, shuffle=True)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
gaussian_pred = gaussian.predict(X_test)
logging.info(f'Gaussian Accuracy: {round((gaussian_pred == y_test).mean(), 3)}')

logistic = LogisticRegression(random_state=42)
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)
logging.info(f'Logistic regression Accuracy: {round((logistic_pred == y_test).mean(), 3)}')

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
logging.info(f'Random Forest Accuracy: {round((rf_pred == y_test).mean(), 3)}')

# ensemble
estimator = []
estimator.append(('Gaussian', GaussianNB()))
estimator.append(('LogisticRegression', LogisticRegression(random_state=42)))
estimator.append(('RF', RandomForestClassifier(random_state=42)))

# Voting Classifier with hard voting
vot_hard = VotingClassifier(estimators=estimator, voting='hard')
vot_hard.fit(X_train, y_train)
voting_pred = vot_hard.predict(X_test)
logging.info(f'Voting Ensemble Accuracy: {round((voting_pred == y_test).mean(), 3)}')

# test on other review
X_test_review = ['口感不错', '口感不错，但送地太慢了', '不太好吃', '味道一般']
X_test_seg = [jieba.lcut(x, cut_all=False) for x in X_test_review]
X_test_vectors = messages_2_vectors(vocabulary_set, X_test_seg)
y_gassian = gaussian.predict(X_test_vectors)
logging.info(f'gaussian: {[REVIEW_DICT[i] for i in y_gassian]}')

y_logistic = logistic.predict(X_test_vectors)
logging.info(f'Logistic regression predict: {[REVIEW_DICT[i] for i in y_logistic]}')

y_rf = rf.predict(X_test_vectors)
logging.info(f'Random Forest predict: {[REVIEW_DICT[i] for i in y_rf]}')

y_voting = vot_hard.predict(X_test_vectors)
logging.info(f'Voting predict: {[REVIEW_DICT[i] for i in y_voting]}')