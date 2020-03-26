import numpy as np
import pandas as pd

# Загружаем данные
train_data = pd.read_csv("data/train.csv", index_col="PassengerId")
test_data = pd.read_csv("data/test.csv", index_col="PassengerId")
print(f"Shape of train data: {train_data.shape}. Shape of test data: {test_data.shape}")

# Удаялем лишние столбцы
columns_to_remove = ['Cabin', 'Ticket']
train_data.drop(columns_to_remove, axis=1, inplace=True)
test_data.drop(columns_to_remove, axis=1, inplace=True)

# Удаляем выбросы

# Заполняем пропущенные значения
train_data['Embarked'].fillna("NAN", inplace=True)
test_data['Embarked'].fillna("NAN", inplace=True) # Можно попробовать использовать вместо NaN моду

mean_age = train_data['Age'].mean()
median_age = train_data['Age'].median()
print(f'Среднее значение возраста: {mean_age}. Медиана: {median_age}')

train_data['Age'].fillna(median_age, inplace=True)
test_data['Age'].fillna(median_age, inplace=True)

mean_fare = test_data['Fare'].mean()
median_fare = test_data['Fare'].median()
print(f'Среднее значение затрат: {mean_fare}. Медиана: {median_fare}')
test_data['Fare'].fillna(mean_fare, inplace=True)

print(f'Пропуски данных в трейн датасете:\n{train_data.isnull().sum()}\n' + "-"*30)
print(f'Пропуски данных в тестовом датасете:\n{test_data.isnull().sum()}\n' + "-"*30)

# Создаем новые характеристики (Feature engineering)
for dataset in [train_data, test_data]:
  # размер семьи
  dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
  dataset['IsAlone'] = 1 #initialize to yes/1 is alone
  dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
  # быстрый и грубый способ отделить титул от имени
  dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

  #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
  #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
  #https://pbpython.com/pandas-qcut-cut.html
  dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

  #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
  dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

# Сохраняем очищенные данные в файл
train_data.to_csv('data/cleaned_train_data.csv', header=True)
test_data.to_csv('data/cleaned_test_data.csv', header=True)