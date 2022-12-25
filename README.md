# Udactiy_DataSc_P2
 Udacity Data Science Project 2 - Disaster Response Pipeline
 
 ## Objectives of Project
 <img width="286" alt="image" src="https://user-images.githubusercontent.com/32632731/209450288-69649292-0b67-43f8-81b7-52aaf3a8d5a6.png">


## Recommended initial Setup & Installations
They are quite some libraries used across the project, all referenced in the requirements.txt file.
For a step by step setup ou can follow those 4 steps:

1. Create a virtual environment
python -m venv .venv

2. Set Execution Policy Settings (optional only if next step leads to error)
Set-ExecutionPolicy Unrestricted -Scope Process
or Set-ExecutionPolicy RemoteSigned

3. Activate venv
.\.venv\Scripts\activate

4. Install required librairies
pip install -r requirements.txt
pip freeze > requirements.txt (to update requirements file with all installed packages)

## How to run the project

1.Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Files Descriptions
Files are splitted into 2 parts:
- preparation, that was used for the planing / testing & experimentation part of the project
- workspace, a replica of the Udacity project folder structure, this is the result of the project work

### Datasets
In the preparation the datasets used are:
-categories.csv, a sample csv file used for ETL pipeline preparation work containing the associated categories for each matching message
-messages.csv, a sample of messages, in original and with its english translation also used for prep. work in the ETL pipeline
-message.db, is the result of the ETL pipeline preparation work, a simple DB with 1 table containing the cleaned messages.

In the workspace folder we can find similar datasets:
-disaster_categories.csv, the file containing the categories for matching messages that are used for the ETL steps.
-disaster_messages.csv, the messages used during the ETL steps.
-DisasterResponse.db, the output of the ETL run data/process_data.py


### Notebooks
There 2 main notebooks & 1 optional notebook in this Repository, there are located in the preparation folder since their sole purpose was to test plan and prepare the different steps in order to be ready to write the python scripts:

- ETL Pipeline Preparation.ipynb, used to load csv datasets, prepare data, clean it and finally save it back to a Table in a Database.

- ML Pipeline Preparation.ipynb, used to load the Database just created from ETL notebook to then train a CLassifier pipeline model and evaluate it. It serves also as testing ground for Cross validation of Model parameters & testing other Models. In the end a selected Model is picked and saved.

- Visuals_Preparation.ipynb, a simple and optional notebook to prepare 2 visualization on the Dataset in order to include them in the web app home page.


### Folders structure

- preparation, contains all testing & experimenting steps of the project.
- workspace, a replica of the Udacity project structure
- workspace/data, all used for the ETL part of the project
- workspace/model, contains the ML pipeline script and the selected model
- workspace/app, contains the html files & scripts to run the web app.

### Python Scripts

- process_data.py, used for ETL
- train_classifier.py, used for ML Pipeline
- run.py, used to start the wep app


## High Level Project Steps

### ETL Pipeline

This Extract Transform Pipeline goes through different fucntions in order to create a clean DB Table from the 2 Csv datasets.

- First it loads and merges message and category data from the specified file paths
- Then, data is cleaned and pre-processed in a this function that performs the following operations on the input dataframe:
        - Splits the 'categories' column into 36 individual columns
        - Renames the columns using the first row of the categories dataframe
        - Converts the values in the category columns to 0 or 1
        - Drops the original 'categories' column
        - Drops duplicates
        - Replace related entries of 2 with 1 (see notebook for more details)
- Finally a last function saves the cleaned data to an SQLite database.

### ML Pipeline

This part of the project is responsible to create a Classifier Model to categorize a message in the selected 36 different ones.

- First it loads data from the given database file.
- There is also a tokenize function that tokenizes and normalizes the given text.
- The Data loaded is splitted between train & test.
- Then in antoher function it builds a model using a pipeline of transformers and a classifier.
- After that it Evaluates the given model on the given test data in another function.
- Finally, it saves the given model to the specified filepath.

#### Model Exploration

The current Model is a RandomForestClassifier using standard parameters, however 3 different Model setup have been explored in the Preparation phase:

- Model #1 - RandomForestClassifier

```python

'clf__estimator__bootstrap': True,
 'clf__estimator__ccp_alpha': 0.0,
 'clf__estimator__class_weight': None,
 'clf__estimator__criterion': 'gini',
 'clf__estimator__max_depth': None,
 'clf__estimator__max_features': 'sqrt',
 'clf__estimator__max_leaf_nodes': None,
 'clf__estimator__max_samples': None,
 'clf__estimator__min_impurity_decrease': 0.0,
 'clf__estimator__min_samples_leaf': 1,
 'clf__estimator__min_samples_split': 2,
 'clf__estimator__min_weight_fraction_leaf': 0.0,
 'clf__estimator__n_estimators': 100,
 'clf__estimator__n_jobs': None,
 'clf__estimator__oob_score': False,
 'clf__estimator__random_state': None,
 'clf__estimator__verbose': 1,
 'clf__estimator__warm_start': False,
 'clf__estimator': RandomForestClassifier(verbose=1),
 'clf__n_jobs': None

```

- Model #2 - RandomForestClassifier, after some help with

Cross Validation Grid Search

```python

parameters = {
    'clf__n_jobs':[2,4,6],
    'clf__estimator__max_depth' : [4,5,6,7,8],
    'clf__estimator__max_features': ['auto','sqrt','log2']
    }
# create grid search object   param_grid=parameters
cv = GridSearchCV(pipeline,param_grid=parameters,verbose=2)

```

the following parameters have been ran:

```python

pipeline.set_params(
    clf__estimator__verbose=1,
    clf__estimator__max_features ='log2',
    clf__n_jobs = 6,
    clf__estimator__n_estimators = 200
    )

```


- Model #3 - KNeighborsClassifier

```python

pipeline_K = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(KNeighborsClassifier())),
])

```

Finally we compared the results:

<img width="467" alt="image" src="https://user-images.githubusercontent.com/32632731/209477577-0f978946-f49b-4fec-8743-9d5fe57c7b7b.png">
<img width="467" alt="image" src="https://user-images.githubusercontent.com/32632731/209477527-a142e367-2c8f-40ba-90a6-0412534eeed7.png">
<img width="467" alt="image" src="https://user-images.githubusercontent.com/32632731/209477514-c4bb1435-413f-4d59-8247-23c51e94ee49.png">
<img width="467" alt="image" src="https://user-images.githubusercontent.com/32632731/209477500-cb7a0714-9293-41c7-95f5-abb88e7719d0.png">


### Web App

This is the final part of the project where it all comes together. The Web app runs a local Flask web app.
From the interface, user can then type in a message and see where the developped model would match this among the 36 categories.
On the Main page you can also see 3 visuals:

- Distribution of Message Genres, part of the sample file given by Udactiy material for the project
- Distribution of Message Categories
![image](https://user-images.githubusercontent.com/32632731/209477841-b7b0ec4b-46d8-40e1-b055-f51fd72c64bf.png)

- Distribution of Message Average Length per Categories
![image](https://user-images.githubusercontent.com/32632731/209477850-075a912f-18d2-4966-8485-c9326170c5f9.png)


## Licensing, Authors, Acknowledgements, etc.

- Data provided by Appen (formally Figure 8) to build a model for an API that classifies disaster messages: https://www.figure-eight.com/
