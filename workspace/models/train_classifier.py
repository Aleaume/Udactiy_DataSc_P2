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

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages_cleaned',engine)

    # X represents the independent variables, or the features, that are used to predict the output variable Y.
    X= df[['message']]

    #Y, on the other hand, represents the dependent variable, or the target, that the model is trying to predict.
    Y = df.drop(columns=['id','message','original','genre'])

    #retrieve category_names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
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

    # build pipeline

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier())),
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    predicted = model.predict(X_test['message'].values.flatten())
    labels = np.unique(predicted)

    i=0
    for col in category_names:

        class_report = classification_report(Y_test[col].to_numpy(), predicted[:,i],labels=labels,output_dict=True)
        sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)

        tn, fp, fn, tp = confusion_matrix(Y_test[col].to_numpy(), predicted[:,i],labels=labels).ravel()
        print("Target: {}\n-----------\nTrue Negative: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Positive: {}\n\n".format(col,tn, fp, fn, tp))
        cm = confusion_matrix(Y_test[col].to_numpy(), predicted[:,i],labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        i+=1
        
        
        
    pass


def save_model(model, model_filepath):
    #saving model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.values.flatten(), Y_train.values)
        
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