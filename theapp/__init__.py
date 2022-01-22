import os
from flask import (
    Flask, g, redirect, render_template, request, url_for, current_app,session
)
import theapp.moviedataframe as movies
import theapp.auth as auth
#import theapp.db as db
# import sqlalchemy as sql
import pandas as pd

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY='yes', DATABASE=os.path.join(
        app.instance_path, 'mydatabase.sqlite'))
    app.config.from_pyfile('config.py', silent=True)
    app.config['TESTING'] = False
    rating_data, movie_data, user_count, movie_count = movies.initialize_movies()
    print(f'user_count: {user_count}, movie_count: {movie_count}')
    app.register_blueprint(auth.bp)

    from . import db
    db.init_app(app)
    with app.app_context():
        db_conn = db.get_db()
    # this can be put into redisd
        db_conn.execute(
            'INSERT INTO config (user_count, movie_count) VALUES (?,?)', (user_count, movie_count)
        )
        movie_data.to_sql('movie_data', db_conn, if_exists='replace', index=False)
        rating_data.to_sql('rating_data', db_conn, if_exists='replace', index=False)
        db_conn.commit()

        # engine = sql.create_engine('sqlite:///mydatabase.sqlite')
        # how can I ensure there are no race conditions?
        
    
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # do we have to import the db after the app has been created?        
    def get_model(and_rest=False):
        db_conn = db.get_db()
        # if in redis:
        #     return redis object

        # if not in redis:
        query = pd.read_sql_query('SELECT * FROM rating_data', db_conn)
        rating_data = pd.DataFrame(query, columns=['user_id', 'movie_id', 'rating']) 
        user_count = db_conn.execute('SELECT * FROM config').fetchone()['user_count']
        movie_count = db_conn.execute('SELECT * FROM config').fetchone()['movie_count']

        print(f'rating_data: {rating_data.head()}')
        print(f'user_count: {user_count}')
        print(f'movie_count: {movie_count}')
            
        model = movies.build_model(rating_data, user_count, movie_count)
        if not and_rest:
            return model
        else:
            return rating_data, movie_data, model

    def get_recommendations(user_id):
        #also implement reddis
        rating_data, movie_data, model = get_model(and_rest=True)
        recommendations = movies.user_recommendations(model, movie_data, rating_data, user_id).to_records()
        return recommendations
    
    def get_similar_movies(movie_id):
        _, movie_data, model = get_model(and_rest=True)
        movie_neighbors = movies.movie_neighbors(model, movie_data, movie_id).to_records()
        return movie_neighbors

    @app.route('/')
    @app.route('/index')
    def index():
        db_conn = db.get_db()
        # query i.e. after ?. in the address
        # If we want to supply query parameters in url_for, we include them as
        # keyword parameters
        query = request.args.get('query', '')
        # here we will query the database instead of going through the pandas
        # so what happens if the query is empty, does it return all results?
        if query:
            movies = db_conn.execute('SELECT * FROM movie_data WHERE title LIKE ?', (query,)).fetchall() #what format is this going to have??
        else:
            movies = db_conn.execute('SELECT * FROM movie_data').fetchall()
        return render_template('index.html', movies=movies) #g.username? how do we store that

    @app.route('/recommendations')
    @auth.login_required
    def recommendations(): 
        recommendations = get_recommendations(session['user_id'])
        return render_template('recommendations.html', movies=recommendations)

        
    @app.route('/userpage')
    @auth.login_required
    def userpage():
        return render_template('userpage.html')

    @app.route('/movie_details/<movie_id>')
    def movie_details(movie_id):
        db_conn =  db.get_db()
        movie_data = db_conn.execute('SELECT * FROM movie_data WHERE movie_id = ?', (movie_id,)).fetchone()
        # similar_movies = get_similar_movies(int(movie_id))
        return render_template('movie_details.html', movie_data=movie_data)
        # return render_template('movie_details.html', movie_data=movie_data, similar_movies=similar_movies.to_records(index=False))

    @app.route('/submit_rating', methods=['POST'])
    @auth.login_required
    def submit_rating():
        #rating data itself will be appended to the rating_data table
        user_id = session['user_id']
        movie_id = request.form['movie_id']
        rating = request.form['rating']

        db_conn = db.get_db()
        print(f'Inserting user_id: {user_id}, movie_id: {movie_id}, rating: {rating}')
        db_conn.execute('INSERT INTO rating_data (user_id , movie_id, rating) VALUES (?,?,?)', (user_id, movie_id, rating))
        db_conn.commit()
        response = db_conn.execute('SELECT * FROM rating_data WHERE (user_id) = ?', (user_id,)).fetchone()
        print('Confirmation:')
        for value in response:
            print(value)

        return redirect(url_for('index'))

    return app
