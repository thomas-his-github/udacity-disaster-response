# import libraries
import sys

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.tree  import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse_table', engine).reset_index()
    
    X = df.message.values
    y = df.loc[:, 'related':'direct_report'].values

    return X, y



def tokenize(text):
    clean_tokens = []

    # Tokenize the string into words
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token.isalpha()]
    clean_tokens.extend(tokens)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
               'clf__estimator__max_depth': [None, 10, 20],
                'clf__estimator__min_samples_split': [2, 5]            
             }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, database_filepath):
    y_pred = model.predict(X_test)

    engine_for_columns = create_engine('sqlite:///' + database_filepath)
    df_for_columns = pd.read_sql_table('DisasterResponse_table', engine_for_columns).reset_index()
    
    columns = df_for_columns.iloc[:, 4:].columns

    for i, category in enumerate(columns):
        print("Category:", category)
        print(classification_report(Y_test[:, i - 1], y_pred[:, i - 1]))
        print("-" * 60)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, database_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
