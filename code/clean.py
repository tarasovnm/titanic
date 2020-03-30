import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

from utils.cleaner import *
from utils.df_transformers import *


# Загружаем данные ============================================================
train_data, test_data = load_data(index_col="PassengerId")

# Удаялем лишние столбцы ======================================================
columns_to_remove = ['Cabin', 'Ticket']

# Удаляем выбросы

# Заполняем пропущенные значения ==============================================
print_na_count(train_data, test_data)

clean_pipeline = make_pipeline(
    RemoveColumnsTransformer(['Cabin', 'Ticket']),
    FillerFunctionTransformer('Embarked', 'value', 'NAN'),
    FillerFunctionTransformer('Age', 'median'),
    FillerFunctionTransformer('Fare', 'mean'),
)

cleaned_train_data = clean_pipeline.fit_transform(train_data)
cleaned_test_data = clean_pipeline.transform(test_data)

print_na_count(cleaned_train_data, cleaned_test_data)

# Создаем новые характеристики (Feature engineering)
print('Feauture engineering =========================================================')
for dataset in [cleaned_train_data, cleaned_test_data]:
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
save_to_csv(cleaned_train_data, cleaned_test_data)