import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# use this row to extract a list of new column names for categories.
def column_part(string):
    '''
    Args: string
    return: first string part
    Takes a string and splits into two parts and returns the first part'''
    first_part = string.split('-')
    return first_part[0]
def number_part(string):
    '''
    Args: string
    return: second string part
    Takes a string and splits into two parts and returns the second part'''
    second_part = string.split('-')
    return second_part[1]

def load_data(messages_filepath, categories_filepath):
    '''
    Args: dataset 1 filepath, dataset 2 filepath
    return: df
    This function reads two csv files and returns a concatenated dataframe. 
    The categories file need to be further split into individual columns and their
    values need to be converted from string to nummeric'''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories)
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0].values.tolist()
    # Applying a lambda function to obtain the column names
    category_colnames = list(map(lambda string: column_part(string), row))
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Converting each category value into numeric
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: number_part(x))
    
        # convert column from string to numeric
        categories[column] =  pd.to_numeric(categories[column]) 
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    return df


def clean_data(df):
    '''
    Args: df
    return: df
    This function drops the duplicates and returns a clean dataset'''
    df = df.drop_duplicates(subset=['message'], keep=False)
    return df


def save_data(df, database_filename):
    '''
    Args: df, database filename
    return: saves the cleaned df in SQLdb
    This function '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseDB', engine,if_exists = 'replace', index=False)
     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()