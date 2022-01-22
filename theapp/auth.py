from werkzeug.security import check_password_hash, generate_password_hash
from theapp.db import get_db
import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db_conn = get_db()
        error = None

        if not username or not password:
            error = 'All fields must be provided'
        # dataframe_user_ids start at 0 so its enough to set the new user id at
        # len(dataframe)
        if not error:
            try:
                config_response = db_conn.execute('SELECT * FROM config').fetchone()
                new_user_id = config_response['user_count']
                movie_count = config_response['movie_count']

            except Exception as e:
                print('FAILED RETIREVING user_count')
                raise e
            print(new_user_id)
            if (new_user_id):
                try:
                    # don't need to retrieve data so cursor not necessary?
                    db_conn.execute(
                        'INSERT INTO user (id, username, password) VALUES (?, ?, ?)',
                        (new_user_id, username, generate_password_hash(password))
                    )
                    db_conn.execute('DELETE FROM config') #but metadata retained?
                    db_conn.execute(
                        'INSERT INTO config (user_count, movie_count) VALUES (?,?)', (new_user_id + 1, movie_count)
                    )
                    db_conn.commit()
                    print('Inserted new user')
                except db_conn.IntegrityError:
                    error = f'User {username} already registered.'
                else:
                    # else is used only when the try succeeds
                    flash('Registration successful, please log in')
                    return redirect(url_for('auth.login'))
        flash(error)
    return render_template('register.html')

@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db_conn = get_db()

        error = None

        if not username or not password:
            error = 'All fields must be provided'  
        else: 
            # surely I should need a cursor here?
            user = db_conn.execute(
                'SELECT * FROM user WHERE username = ?', (username,)
            ).fetchone()
            if user:
                if not check_password_hash(user['password'], password):
                    error = 'Incorrect password'
                else:
                    session.clear()
                    session['user_id'] = user['id']
                    session['username'] = user['username']

                    if request.args.get('redirect'):
                        return redirect(url_for(request.args.get('redirect')))
                    else:
                        return redirect(url_for('index'))
            else:
                error = 'No user found'
            flash(error)
    return render_template('login.html')

@bp.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('index'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'username' not in session:
            # should return an additional page saying the user is not logged in
            return redirect(url_for('auth.login', redirect=view.__name__))
        return view(**kwargs)
    return wrapped_view