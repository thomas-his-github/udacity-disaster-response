import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from messages and category files and merge into a single dataframe.

    Args:
    messages_filepath : Filepath for the messages CSV file.
    categories_filepath : Filepath for the categories CSV file.

    Returns:
    df : Merged dataframe containing messages and categories.
    """
    # read messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge to dataframe and return
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df, categories_filepath):
    """
    Clean the merged dataframe

    Args:
    df : Merged dataframe containing messages and categories.
    categories_filepath : Filepath for the categories CSV file.

    Returns:
    df_clean : Cleaned dataframe 
    """
    categories = pd.read_csv(categories_filepath)
    categories = categories['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories
    category_colnames = row.map(lambda x: x.split('-')[0]).tolist()

    categories.columns = category_colnames

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # Replace wrong values of '2' by '1'
        categories[column] = categories[column].replace('2', '1')
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)

    df_clean = df.drop_duplicates()

    return df_clean


def save_data(df, database_filename):
    """
    Save the cleaned dataframe to a database.

    Args:
    df : Cleaned dataframe.
    database_filename : Filepath for the database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')


def main():
    """
    Main function to load data, clean data, and save data to a database.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories_filepath)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database! Great work!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
