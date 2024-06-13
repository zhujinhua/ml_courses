"""
Author: jhzhu
Date: 2024/6/13
Description: 
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
joblib.dump(rf, filename='iris.joblib')