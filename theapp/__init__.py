import os 
from flask import (
    Flask, g, redirect, render_template, request, url_for
)
from theapp.database import *

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY=True)
    app.config.from_pyfile('config.py',silent=True)
    try:
        os.makedirs(app.instance_path) 
    except OSError:
        pass
    

    @app.route('/')
    def main():
        query = request.args.get('query', '')
        if query:
            items = find_movies(item_data,query)
            return render_template('all.html', items=items.to_records(index=False), user=current_user)
        else:
            return render_template('all.html', items=item_data.to_records(index=False), user=current_user)
        
    
    @app.route('/recommendations')
    def recommendations():
        if current_user:
            recommendations = user_recommendations(model, current_user)
            return render_template('recommendations.html', items=recommendations.to_records(), user=current_user)
        else:
            return redirect(url_for('userpage.html'))
        

    @app.route('/item_details/<item_id>')
    def item_details(item_id):
        item_title=item_data[item_data['movie_id']==str(item_id)]['title'].to_string(index=False)
        #list(data_set.itertuples(index=False, name=None))
        similar_movies=movie_neighbors(model,int(item_id))
        return render_template('item_details.html', title=item_title, item_id=item_id, items=similar_movies.to_records(index=False))
    
    @app.route('/submit_rating')
    def submit_rating():
        movie_id = request.args.get('movie_id','')
        rating = request.args.get('rating','')
        insert_new_rating(rating_data, current_user, movie_id, rating)
        return redirect(url_for('main'))

    @app.route('/userpage')
    def userpage():
        return render_template('userpage.html', user=current_user)

    @app.route('/create_user')
    def create_user():
        global current_user
        current_user = int(create_new_user())
        return redirect(url_for('main'))

    @app.route('/change_user', methods=['POST'])
    def change_user():
        global current_user
        user_id = request.form['userchoice']
        if user_id in user_list:
            current_user = int(user_id)
        return redirect(url_for('main'))
    return app