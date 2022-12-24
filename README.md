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

##Files Descriptions

###Datasets

###Notebooks

###Folders structure

###Python Scripts


##Licensing, Authors, Acknowledgements, etc.
