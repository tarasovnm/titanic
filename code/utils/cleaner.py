import pandas as pd

class Cleaner():
  def __init__(self):
    self.data = dict()


  def load_data(self, index_col=None):
    print('Loading data =================================================================')
    self.data['train'] = pd.read_csv("../data/input/train.csv", index_col=index_col)
    self.data['test'] = pd.read_csv("../data/input/test.csv", index_col=index_col)
    print('Data loaded')
    print(f"Shape of train data: {self.data['train'].shape}. Shape of test data: {self.data['test'].shape}")


  def remove_columns(self, columns_list):
    print('Delete columns: ==============================================================')
    self.data['train'].drop(columns_list, axis=1, inplace=True)
    self.data['test'].drop(columns_list, axis=1, inplace=True)
    print(f'Cтолбцы удалены: {columns_list}')

  
  def save_to_csv(self, header=True):
    print('Saving data to CSV ===========================================================')
    self.data['train'].to_csv('../data/working/cleaned_train_data.csv', header=header)
    self.data['test'].to_csv('../data/working/cleaned_test_data.csv', header=header)
    print('Files saved')


  def fill_na(self, columns_values_dict):
    print('Filling missing values =======================================================')
    for column in columns_values_dict:
      self.data['train'][column].fillna(columns_values_dict[column], inplace=True)
      self.data['test'][column].fillna(columns_values_dict[column], inplace=True)
      print(f'Пропуски в столбце {column} заполнены значением {columns_values_dict[column]}')

    train_data = self.data['train']
    test_data = self.data['test']
    print(f'Пропуски данных в трейн датасете:\n{train_data.isnull().sum()}\n' + "-"*40)
    print(f'Пропуски данных в тестовом датасете:\n{test_data.isnull().sum()}\n' + "-"*40)

