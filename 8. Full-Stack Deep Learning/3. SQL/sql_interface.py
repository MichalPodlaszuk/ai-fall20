import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('ml_app.db')
except Exception as ex:
    print(ex)

def sql_query(sentence):
    try:
        c = conn.cursor()
        reply = c.execute(sentence)
        conn.commit()
        return reply
    except Exception as ex:
        print(ex)
        return ex

def pandas_select(sentence):
    try:
        if sentence.split()[0].lower() == 'select':
            df = pd.read_sql_query(sentence, conn)
            return df['email']
        else:
            return pd.DataFrame()
    except Exception as ex:
        print(ex)
        return pd.DataFrame()


def upload_data(name: str, path, head=0):
    try:
        df = pd.read_csv(path)
        frame = df.to_sql(name, conn, if_exists='replace')
    except Exception as ex:
        print(ex)


