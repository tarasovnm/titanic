import numpy as np
import pandas as pd
from sklearn import pipeline, preprocessing
from sklearn.pipeline import Pipeline, make_pipeline

from utils.df_transformers import *

# Загружаем данные ===========================================================
train_data = pd.read_csv("data/cleaned_train_data.csv", index_col="PassengerId")
test_data = pd.read_csv("data/cleaned_test_data.csv", index_col="PassengerId")
print(f"Shape of train data: {train_data.shape}. Shape of test data: {test_data.shape}")
train_size = train_data.shape[0]

# Объединим датасеты для обработки ===========================================
data = pd.concat([train_data.drop(['Survived'], axis=1), test_data])
y = train_data['Survived']

# Разделяем признаки по типу =================================================
# Бинарные
# Числовые
num_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
# Категориальные
cat_features = train_data.columns.drop(num_features).drop(['Survived', 'Name']).tolist()
# Текстовые
text_features = ['Name']

print(f'Числовые признаки: {num_features}')
print(f'Номинальные признаки: {cat_features}')
print(f'Текстовые признаки: {text_features}')

# Собираем пайплайн ==========================================================
# Обработка числовых признаков
num_pipeline = make_pipeline(
    SelectColumnsTransfomer(num_features),
    #preprocessing.StandardScaler(with_mean = 0)
)

# Обработка категориальных признаков
cat_pipeline = make_pipeline(
    SelectColumnsTransfomer(cat_features),
    DataFrameFunctionTransformer(lambda x: x.apply(str)),
    # Разобраться что за тип данных category в pd.DataFrame и что такое object_levels
    #DataFrameFunctionTransformer(lambda x:x.astype('category', categories=object_levels)),
    ToDummiesTransformer(),
)

# Обединяем обработанные данные
preprocessing_features = DataFrameFeatureUnion([num_pipeline, cat_pipeline])

# Обрабатываем и разделяем данные
prprd_data = preprocessing_features.fit_transform(data)
preprocessed_train_data = prprd_data.iloc[:train_size, :]
preprocessed_test_data = prprd_data.iloc[train_size:, :]
preprocessed_train_data['Survived'] = y

# =============================================================================
preprocessed_train_data.to_csv('data/preprocessed_train_data.csv', header=True)
preprocessed_test_data.to_csv('data/preprocessed_test_data.csv', header=True)