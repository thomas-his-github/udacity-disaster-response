# udacity-disaster-response

## Summary
Contains the final project for the data engineering course.

In this project a dataset is used with labeled messages that were sent during disaster events. The scripts in this project clean these messages and build a machine learning pipeline to classify new messages. The output will be a web application where these new messages can be classified.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files used in this project

### Scripts
**run.py** - Script that executes the web application (HTML files in templates folder are used by this script).<br/>
**process_data.py** - Script that loads, cleans and creates the features based on the disaster messages, and stores this in a database.<br/>
**train_classifier.py** - Script that contains the machine learning pipeline to train based on the messages database and creates a pickle file used for future prediction.<br/>

### Data

**disaster_messages.csv** - Messages sent during disaster events<br/>
**disaster_categories.csv** - Categories of the disaster messages<br/>
