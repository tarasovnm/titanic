import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import model_selection, metrics, feature_selection
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.ensemble import GradientBoostingClassifier


# Загружаем данные ======================================================================
train_data = pd.read_csv("../data/output/preprocessed_train_data.csv", index_col="PassengerId")
test_data = pd.read_csv("../data/output/preprocessed_test_data.csv", index_col="PassengerId")
print(f"Shape of train data: {train_data.shape}. Shape of test data: {test_data.shape}")

# Разделяем выборку на трейн и тест =====================================================
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(f'Shape of data before feature selection: {X_train.shape}')

# model
cv = KFold(n_splits=7, random_state=0)

def accuracy_cv(model):
  kfold = cv
  result = dict()
  result['train_score'] = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
   
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  result['test_score'] = metrics.accuracy_score(y_test, predictions)
  return result

gbc = GradientBoostingClassifier()

cv_results = accuracy_cv(gbc)
print(f'Результаты работы классификатора до отбора фич: на трейне {cv_results["train_score"].mean()}, на тесте {cv_results["test_score"]}')

# feature selection
dtree_rfe = feature_selection.RFECV(gbc, step = 1, scoring = 'accuracy', cv = cv)
dtree_rfe.fit(X_train, y_train)
X_train_rfe = dtree_rfe.transform(X_train)
X_test_rfe = dtree_rfe.transform(X_test)
X_rfe = dtree_rfe.transform(X)
test_data_rfe = dtree_rfe.transform(test_data)

print(f'Shape of transformed data: {X_train_rfe.shape}')
gbc.fit(X_train_rfe, y_train)
predictions = gbc.predict(X_test_rfe)
print(f'Значение accuracy для gbc на тестовых данных после отбора фич: {metrics.accuracy_score(y_test, predictions)}')

gbc.fit(X_rfe, y)
predicted = gbc.predict(test_data_rfe)
test_data["Survived"] = predicted
test_data["Survived"].to_csv('../submissions/gbc_06_04_2020v5.csv', header=True)
print('Сабмишн успешно записан на диск')