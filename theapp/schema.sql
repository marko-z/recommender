DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS rating_data;
DROP TABLE IF EXISTS movie_data;
DROP TABLE IF EXISTS config;

CREATE TABLE user (
    id INTEGER PRIMARY KEY, -- will be compatible with the ids of the users in the dataframe?
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);


CREATE TABLE rating_data (
    id INTEGER PRIMARY KEY, -- will be compatible with the ids of the users in the dataframe?
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating NUMBER(10)
);

CREATE TABLE movie_data (
    id INTEGER PRIMARY KEY, -- again ID taken from the pandas dataframe
    title VARCHAR(255) NOT NULL,
    avg_rating NUMBER(10) NOT NULL,
    genre VARCHAR(255) NOT NULL
);

CREATE TABLE config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_count INTEGER,
    user_count INTEGER
);