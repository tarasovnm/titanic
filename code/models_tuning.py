import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import model_selection, metrics
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

# Определим стратегию кроссвалидации ====================================================
seed = 3
n_folds = 7

def accuracy_cv(model):
  kfold = KFold(n_splits=n_folds, random_state=seed)
  result = dict()
  result['train_score'] = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
   
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  result['test_score'] = metrics.accuracy_score(y_test, predictions)
  return result

# Лучшее значение дает GradientBoostingClassifier =======================================
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
predictions = gbc.predict(X_test)
print(f'Значение accuracy для GradientBoostingClassifier на тестовых данных без тюнинга гиперпараметров: {metrics.accuracy_score(y_test, predictions)}')

# Поиск параметров по сетке
# parameters_grid = {
#   'loss': ['deviance', 'exponential'],
#   'learning_rate': np.linspace(0.01, 0.2, 10, endpoint=True),
#   'n_estimators': [100, 150, 200, 250],
#   'max_depth': [1, 2, 3, 4, 5],
#   'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
#   'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
#   'max_features': list(range(1,X_train.shape[1])),
# }

# cv = KFold(n_splits=n_folds, random_state=seed)
# grid_cv = model_selection.RandomizedSearchCV(gbc, parameters_grid, scoring='accuracy', cv=cv, n_iter=1000)
# grid_cv.fit(X_train, y_train)

# print(grid_cv.best_estimator_)
# print(grid_cv.best_score_)
# print(grid_cv.best_params_)

# Best parameters
# {'n_estimators': 200, 'min_samples_split': 0.1, 'min_samples_leaf': 0.1, 'max_features': 16, 'max_depth': 4, 'loss': 'deviance', 'learning_rate': 0.1366666666666667}

gbc_best = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           learning_rate=0.1366666666666667, loss='deviance',
                           max_depth=4, max_features=16, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=0.1, min_samples_split=0.1,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='auto',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)

gbc_best.fit(X_train, y_train)
predictions_best = gbc_best.predict(X_test)
print(f'Значение accuracy для GradientBoostingClassifier на тестовых данных после тюнинга: {metrics.accuracy_score(y_test, predictions_best)}')

# Создаем сабмишн =======================================================================
gbc_best.fit(X, y)
predicted = gbc_best.predict(test_data)
test_data["Survived"] = predicted
test_data["Survived"].to_csv('../submissions/gbc_03_04_2020v2.csv', header=True)
print('Сабмишн успешно записан на диск')