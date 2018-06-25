import numpy as np
import pandas as pd
import os

def read_data():
    #set the path of the raw data
    raw_data_path = os.path.join(os.pardir, 'data', 'raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    
    train_df = pd.read_csv(train_file_path, index_col = 'PassengerId')
    test_df = pd.read_csv(test_file_path, index_col = 'PassengerId')
    test_df['Survived'] = -888
    
    df = pd.concat([train_df, test_df], axis=0)
    return df

def process_data(df):
    # using method chaining
    # adding the title column
    df = df.assign(Title = df.Name.apply(get_title))
            
    

    # fill missing values
    df = df.pipe(fill_missing_values)
    # create fare bin feature 
    df = df.assign(Fare_bin = pd.qcut(df.Fare,4,labels = ['very_low', 'low', 'high', 'very_high']))\
    .assign(AgeState = np.where(df.Age >= 18, 'Adult', 'Child'))\
    .assign(FamilySize = df.Parch + df.SibSp + 1)\
    .assign(IsMother =  np.where((df.Sex == 'F') & (df.Parch > 0) & (df.Age >18) & (df.Title != 'Miss'),1,0))\
    .assign(Cabin = np.where(df.Cabin == 'T', np.nan, df.Cabin))
    df = df.assign(Deck = df.Cabin.apply(get_deck))\
    .assign(IsMale = np.where(df.Sex=='male',1,0 ))\
    .pipe(pd.get_dummies, columns = ['Deck', 'Pclass', 'Title', 'Fare_bin', 'Embarked', 'AgeState'])\
    .drop(['Cabin', 'Name','Ticket', 'Parch', 'SibSp', 'Sex'], axis = 1)\
    .pipe(reorder_columns)
    return df
           
def get_title(name):
    title_group = {
        'mr': 'Mr',
        'mrs': 'Mrs',
        'miss': 'Miss',
        'master': 'Master',
        'don': 'Sir',
        'rev': 'Sir',
        'dr': 'Officer',
        'lady': 'Lady',
        'major': 'Officer',
        'mme': 'Mrs',
        'ms': 'Mrs',
        'sir': 'Sir',
        'mlle': 'Miss',
        'col': 'Officer',
        'capt': 'Officer',
        'the countess': 'Lady',
        'jonkheer': 'Sir',
        'dona': 'Lady'   
    }
    title_part = name.split(',')[1]
    title = title_part.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def fill_missing_values(df):
    # embarked
    df.Embarked.fillna('C', inplace=True)
    # fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')].Fare.median()
    df.Fare.fillna(median_fare, inplace = True)
    # age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace = True)
    return df

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df

def write_data(df):
    processed_data_pata = os.path.join(os.pardir, 'data', 'processed')
    write_train_path = os.path.join(processed_data_pata, 'train.csv')
    write_test_path = os.path.join(processed_data_pata, 'test.csv')
    # train data
    df[df.Survived != -888].to_csv(write_train_path)
    # test
    columns = [column for column in df.columns if column != 'Survived']
    df[df.Survived == -888][columns].to_csv(write_test_path)
    
if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)