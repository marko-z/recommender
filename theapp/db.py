import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext

# make db connection if none and return 
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        g.db.row_factory = sqlite3.Row
    return g.db

# check if db in g and remove it if it is
def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


# retrieve the db object from g and use it to populate the database according to
# file 'schema.sql'
def init_db():
    db_conn = get_db()
    print('Executing init_db()...')
    with current_app.open_resource('schema.sql') as f:
        print('Reading schema.sql')
        db_conn.executescript(f.read().decode('utf-8'))

# same as above but as a cli command 'flask init-db'
@click.command('init-db')
@with_appcontext
def init_db_command():
    init_db()
    click.echo('Initialized database')
    

# to be executed in the __init__.py file as the app instance is created
def init_app(app):
    print('Executing init_app')
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
