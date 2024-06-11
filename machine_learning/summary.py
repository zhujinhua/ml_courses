import time

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# streamlit

def clock(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        time_cost = end_time - start_time
        print(f"Time cost for {func.__name__}: {time_cost:.6f} seconds")
        return result

    return wrapper


X, y = make_classification(n_samples=5000, n_features=100, n_informative=40,
                           n_redundant=5, n_classes=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


@clock
def knn_classifier():
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('KNN Accuracy: %s' % (y_pred == y_test).mean())


@clock
def rf_classifier():
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Random Forest Accuracy: %s' % (y_pred == y_test).mean())


knn_classifier()
rf_classifier()
