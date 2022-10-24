# import libraries
import sys
import pandas as pd
import pickle
import numpy as np
import re
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier as clf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''
    Args: database_filepath
    return: Training variable(as numpy array), target variables(df), target names as list
    This function obtain the data from the database and returns X,y and the list of y names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df.message
    y = df.iloc[:, 4:]
    # y=y.astype('int')
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    Args: text
    return: returns tokens(list)
    text: This function takes a text, then normalize it, tokenize it, lemmatize it and returns a list containing these
    tokens
    '''
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        tokens.append(clean_tok)
    return tokens


def build_model():
    '''
    Args: -
    return: returns the model
    This function uses a pipeline, takes the parameters and build the model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', clf(RandomForestClassifier()))

    ])
    parameters = {'clf__estimator__n_estimators': [10],
                  'clf__estimator__min_samples_split': [2]
                  }

    model = GridSearchCV(pipeline, param_grid=parameters)
    # cv.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
       Args: model, testing variables (X and Y) and the category names
       return: Prints the classification report
       This function evaluates the model and predicts for the test set. A classification report giving the f1 score,
       precision and recall is printed out
      '''

    Y_pred = model.predict(X_test)
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i], zero_divsion=0, target_names=category_names),'...............................................')


def save_model(model, model_filepath):
    '''
    Args: model and model file path
    return: Outputs a pickle file of the trained model
    This function creates a pickle file for the trained model.
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print("\nBest Parameters:", model.best_params_)

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