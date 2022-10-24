# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### File Structure:
1. app Folder:
   1. templates folder
   2. run.py to run the app
2. data Folder:
   1. categories and messages csv files
   2. ETL database
   3. pre-processed python file
3. models
   1. classification model file
4. Notebooks
   1. These contain the ETL and ML preparation notebooks (not needed for running the app)
