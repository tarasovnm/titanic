import numpy as np
import pandas as pd
from utils.cleaner import Cleaner

cleaner = Cleaner()

# Загружаем данные
cleaner.load_data(index_col="PassengerId")

# Удаялем лишние столбцы
columns_to_remove = ['Cabin', 'Ticket']
cleaner.remove_columns(columns_to_remove)

# Удаляем выбросы

# Заполняем пропущенные значения
mean_age = cleaner.data['train']['Age'].mean()
median_age = cleaner.data['train']['Age'].median()

mean_fare = cleaner.data['test']['Fare'].mean()
median_fare = cleaner.data['test']['Fare'].median()

values_dict = {'Embarked': 'NAN',
              'Age': median_age,
              'Fare': mean_fare}

cleaner.fill_na(values_dict)

# Создаем новые характеристики (Feature engineering)
print('Feauture engineering =========================================================')
for dataset in [cleaner.data['train'], cleaner.data['test']]:
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
cleaner.save_to_csv()