import numpy as np
import pandas as pd

from sklearn import pipeline, preprocessing, ensemble, linear_model, svm, tree
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, ShuffleSplit, cross_validate
from sklearn import model_selection, metrics

#import models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# from utils.df_transformers import SelectColumnsTransformer


# Загружаем данные ======================================================================
train_data = pd.read_csv("../data/output/preprocessed_train_data.csv", index_col="PassengerId")
test_data = pd.read_csv("../data/output/preprocessed_test_data.csv", index_col="PassengerId")
print(f"Shape of train data: {train_data.shape}. Shape of test data: {test_data.shape}")

X = np.array(train_data.drop(['Survived'], axis=1))
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

# Модели ================================================================================
models = [
  #Ensemble Methods
  ensemble.AdaBoostClassifier(),
  ensemble.BaggingClassifier(),
  ensemble.ExtraTreesClassifier(),
  ensemble.GradientBoostingClassifier(),
  ensemble.RandomForestClassifier(),

  #Gaussian Processes
  #gaussian_process.GaussianProcessClassifier(),
    
  #GLM
  linear_model.LogisticRegressionCV(),
  linear_model.PassiveAggressiveClassifier(),
  linear_model.RidgeClassifierCV(),
  linear_model.SGDClassifier(),
  linear_model.Perceptron(),
    
  #Navies Bayes
  #naive_bayes.BernoulliNB(),
  #naive_bayes.GaussianNB(),
    
  #Nearest Neighbor
  #neighbors.KNeighborsClassifier(),
    
  #SVM
  svm.SVC(probability=True),
  svm.NuSVC(probability=True),
  svm.LinearSVC(),
    
  #Trees    
  tree.DecisionTreeClassifier(),
  tree.ExtraTreeClassifier(),
    
  #Discriminant Analysis
  #discriminant_analysis.LinearDiscriminantAnalysis(),
  #discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
  #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
  XGBClassifier()    
  ]

  # Создадим таблицу для сравнения метрик алгоритмов
MLA_columns = ['MLA Name', 'MLA Train Accuracy Mean', 'MLA Train Accuracy 3*STD']
MLA_compare = pd.DataFrame(columns = MLA_columns)

# Создадим таблицу для сравнения предсказаний алгоритмов
MLA_predict = train_data['Survived']

row_index = 0
for model in models:
  MLA_name = model.__class__.__name__
  MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
   
  cv_results = accuracy_cv(model)
    
  MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
  MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
  #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
  MLA_compare.loc[row_index, 'MLA Train Accuracy 3*STD'] = cv_results['train_score'].std()*3   #let's know the worst that can happen!
    
  #save MLA predictions - see section 6 for usage
  #model.fit(X, y)
  #MLA_predict[MLA_name] = model.predict(X)
    
  row_index+=1

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print(MLA_compare)

# Лучшее значение дает GradientBoostingClassifier =======================================
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
predictions = gbc.predict(X_test)
print(f'Значение accuracy для GradientBoostingClassifier: {metrics.accuracy_score(y_test, predictions)}')

# Создаем сабмишн =======================================================================
gbc.fit(train_data.drop(['Survived'], axis=1), y)
predicted = gbc.predict(test_data)
test_data["Survived"] = predicted
test_data["Survived"].to_csv('../submissions/gbc_03_04_2020v1.csv', header=True)
print('Сабмишн успешно записан на диск')
