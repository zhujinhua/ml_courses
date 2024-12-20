from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import logging
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)


def get_entropy(logits):
    logits = np.array(logits)
    if len(logits) < 2:
        return 0
    probs = np.array([(logits == label).mean() for label in set(logits)])
    entropy = - (probs * np.log2(probs)).sum()
    # gini = probs * (1 - probs)
    return entropy


def calculate_entropy(probabilities):
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
logging.info(accuracy_score(y_test, tree_pred))
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
plot_tree(tree, filled=True)
plt.show()

best_entropy = float('inf')
best_feature_idx = None
best_feature_value = None

for feature_index in range(np.shape(X_train)[1]):
    logging.info(f'feature index: {feature_index} {sorted(set(X_train[:, feature_index]))}')
    for value in set(X_train[:, feature_index]):
        # logging.info(f'feature: {feature_index}')
        # logging.info(f'value: {value}')
        y_left = y_train[X_train[:, feature_index] <= value]
        y_right = y_train[X_train[:, feature_index] > value]
        left_entropy = get_entropy(y_left)
        right_entropy = get_entropy(y_right)
        length = len(X_train[:, feature_index])
        all_entropy = left_entropy * len(y_left) / length + right_entropy * len(y_right) / length
        if all_entropy < best_entropy:
            best_entropy = all_entropy
            best_feature_idx = feature_index
            best_feature_value = value
logging.info(f'best split point: {best_feature_value}, {best_feature_idx}, {best_entropy}')
