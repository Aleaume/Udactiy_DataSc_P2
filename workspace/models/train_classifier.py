import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay

def load_data(database_filepath):
    """Load data from the given database file.

    Parameters:
    - database_filepath (str): the filepath of the database to load the data from.

    Returns:
    - X (np.ndarray): an array of independent variables, or features, that are used to predict the output variable Y.
    - Y (pd.DataFrame): a dataframe of dependent variables, or targets, that the model is trying to predict.
    - category_names (List[str]): a list of column names for the dependent variables in Y.
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages_cleaned',engine)

    # X represents the independent variables, or the features, that are used to predict the output variable Y.
    X = df['message'].values

    #Y, on the other hand, represents the dependent variable, or the target, that the model is trying to predict.
    Y = df.drop(columns=['id','message','original','genre'])

    #retrieve category_names
    category_names = df.drop(columns=['id','message','original','genre']).columns

    return X, Y, category_names


def tokenize(text):
    """Tokenize and normalize the given text.

    Parameters:
    - text (str): the text to tokenize and normalize.

    Returns:
    - A list of tokens (str) after normalizing, tokenizing, removing stop words, reducing to stems, and reducing to root form.
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their stems
    words = [PorterStemmer().stem(w) for w in words]

    # Reduce words to their root form
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words


def build_model():
    """Build a model using a pipeline of transformers and a classifier.

        It creates:
                - A pipeline object that includes CountVectorizer, TfidfTransformer, and a MultiOutputClassifier with a RandomForestClassifier.
                - Hyperparameters to cross-validate
                - GridSearch
    Returns:
    - A model object that includes a GridSearch Cross-validation based on selected Hyperparameters.
    """

    # build pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier())),
    ])

    #setting hyperparameters to cross-validate
    parameters = {
        'clf__n_jobs':[2,4,6],
        'clf__estimator__max_depth' : [4,8],
        'clf__estimator__max_features': ['auto','sqrt','log2']
        }

    # create grid search object   param_grid=parameters
    cv = GridSearchCV(pipeline,param_grid=parameters,verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """Evaluate the given model on the given test data.

    This function generates a classification report and confusion matrix for each category in the test data. The classification report includes precision, recall, f1-score, and support for each label. The confusion matrix includes true positive, false positive, false negative, and true negative counts for each label.

    Parameters:
    - model (Pipeline): the model to evaluate.
    - X_test (np.ndarray): the independent variables, or features, for the test data.
    - Y_test (pd.DataFrame): the dependent variables, or targets, for the test data.
    - category_names (List[str]): the names of the categories in Y_test.

    Returns:
    - None
    """

    predicted = model.predict(X_test)
    labels = np.unique(predicted)

    i=0
    for col in category_names:

        class_report = classification_report(Y_test[col].to_numpy(), predicted[:,i],labels=labels,output_dict=True)
        tn, fp, fn, tp = confusion_matrix(Y_test[col].to_numpy(), predicted[:,i],labels=labels).ravel()
        print("Target: {}\n-----------\nTrue Negative: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Positive: {}\n\n".format(col,tn, fp, fn, tp))
        print(class_report)

        i+=1
        
        
        
    pass


def save_model(model, model_filepath):
    """Save the given model to the specified filepath.

    Parameters:
    - model (Any): the model to save.
    - model_filepath (str): the filepath to save the model to.

    Returns:
    - None
    """
    #saving model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    """Train a model on disaster response messages data and save it to a specified filepath.

    This function loads data from a database, splits it into train and test sets, builds and trains a model, evaluates the model, and then saves the model to a specified filepath.

    The filepaths for the database and the pickle file to save the model to should be provided as command line arguments in the following order: database filepath, model filepath.

    If the required number of arguments is not provided, the function will print a usage message.
    """


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

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()