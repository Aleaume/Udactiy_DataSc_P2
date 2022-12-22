import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge message and category data from the specified file paths.
    
    Args:
        messages_filepath (str): The file path for the messages data.
        categories_filepath (str): The file path for the categories data.
    
    Returns:
        pd.DataFrame: The merged dataframe containing both the messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on=['id'])

    return df


def clean_data(df):
    """Clean and preprocess the input dataframe.
    
    This function performs the following operations on the input dataframe:
        - Splits the 'categories' column into 36 individual columns
        - Renames the columns using the first row of the categories dataframe
        - Converts the values in the category columns to 0 or 1
        - Drops the original 'categories' column
        - Drops duplicates
        - Replace related entries of 2 with 1 (see notebook for more details)
    
    Args:
        df (pd.DataFrame): The input dataframe containing the raw messages and categories data.
    
    Returns:
        pd.DataFrame: The cleaned and preprocessed dataframe.
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)

    # select the first row of the categories dataframe
    row = categories[:1]

    # extract a list of new column names for categories.
    category_colnames = (row.apply(lambda x : x.str[:-2],axis=1)).values.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].apply(lambda x : x[-1:])
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)

    # Convert `categories` columns from MultiIndex to Index
    categories.columns = categories.columns.get_level_values(0)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    
    #clean related column, replace 2  with 1

    df.related = df.related.replace(2,1,inplace=True)

    return df


def save_data(df, database_filename):
    """Save the cleaned data to an SQLite database.
    
    Args:
        df (pd.DataFrame): The cleaned dataframe to save.
        database_filename (str): The filepath for the SQLite database.
    
    Returns:
        None
    """
    #Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages_cleaned', engine, index=False)

    pass  


def main():

    """Load, clean, and save data to an SQLite database.
    
    This function expects the filepaths for the messages and categories data as the first and second command line arguments, respectively, and the filepath for the SQLite database to save the cleaned data to as the third argument. It loads the data using the `load_data()` function, cleans it using the `clean_data()` function, and saves it to the specified database using the `save_data()` function.
    
    Returns:
        None
    """

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