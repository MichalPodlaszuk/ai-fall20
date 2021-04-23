import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import os
import pymysql

conn = sqlite3.connect("strive.db")


def get_remote_conn(user, pas, IP, port):
    try:
        engine = create_engine("mysql+pymysql://{}:{}@{}/{}?host={}?port={}").format(user, pas, IP, user, IP, port)
        conn = engine.connect()
        return conn
    except Exception as ex:
        print(ex)
        return ex


def sql_execute(sentence):
    try:
        c = conn.cursor()
        a = c.execute(sentence)
        conn.commit()
        print(a)
    except Exception as ex:
        print(ex)
        return ex


def pd_select(sentence):
    try:

        df = pd.read_sql_query(sentence, conn)
        return df
    except Exception as ex:
        print(ex)
        return ex


def pd_upload_csv(name: str, pth, head=0):
    try:
        df = pd.read_csv(pth, header=head)
        frame = df.to_sql(name, conn, if_exists='replace')
        return frame
    except Exception as ex:
        print(ex)
        return ex


def close():
    conn.close()


create = "create table student(name text)"
insert = "insert into student(name) Values('jon')"
drop = "drop table student"
a = sql_execute(insert)
df = pd_select("select * from student")
print(df)