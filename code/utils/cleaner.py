import pandas as pd


def load_data(index_col=None):
  print('Loading data =================================================================')
  train_data = pd.read_csv("../data/input/train.csv", index_col=index_col)
  test_data = pd.read_csv("../data/input/test.csv", index_col=index_col)
  print('Data loaded')
  print(f"Shape of train data: {train_data.shape}. Shape of test data: {test_data.shape}")

  return train_data, test_data


def print_na_count(train_data, test_data):
  print(f'Пропуски данных в трейн датасете:\n{train_data.isnull().sum()}\n' + "-"*40)
  print(f'Пропуски данных в тестовом датасете:\n{test_data.isnull().sum()}\n' + "-"*40)


def save_to_csv(train_data, test_data, header=True):
    print('Saving data to CSV ===========================================================')
    train_data.to_csv('../data/working/cleaned_train_data.csv', header=header)
    test_data.to_csv('../data/working/cleaned_test_data.csv', header=header)
    print('Files saved')