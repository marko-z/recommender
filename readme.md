# Movies 4 You (movie recommender)

## Description

Movie recommender built on top of the MovieLens dataset, using flask/jinja2 as backend, with tensorflow for matrix reduction when determining similary between users and items. Uses sqlite3 for handling user registration and persistent storage of user ratings and movie data.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage 

Inside the base `recommender/` directory:
``` 
FLASK_APP=theapp flask init-db
FLASK_ENV=development FLASK_APP=theapp flask run
```
navigate to the localhost address.
