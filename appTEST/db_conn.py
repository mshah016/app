import tensorflow as tf
from tensorflow import keras
from model import Model

import pymysql
import pandas as pd
from params import mysql_pass


def getData():
    """gets data from database and returns list of lists containing user data"""

    # establish connection
    conn = pymysql.connect(
	host='ometrics.c8llxlpivoxb.us-east-1.rds.amazonaws.com',
	user='root',
	password=mysql_pass,
	db='ometrics',
        ssl_disabled= True
        )

    cur = conn.cursor()

    # select query
    columns = 'id, chat_time, user_id, chat_context, is_user_response, user_response'
    table = 'chat_tracker_test'
    conditions = 'WHERE chat_time LIKE "0000-00-00%" AND is_button_click = "0" AND is_user_response = "1"'
    query = f'Select {columns} from {table} {conditions};'

    # convert retrieved data into dataframe 
    df = pd.read_sql_query(query, con = conn)

    return conn, cur, df



def insertValues(): 
    """calls getData function retrieve dataframe and append to table""" 
    
    # retrieve connection engine and dataframe from getData()
    conn, cur, df = getData()

    # retrieve sent_list from Model()
    _, sent_list, sent_score_list = Model(df['user_response'])
    #sent_score_list = Model(df['user_response'])

    # append sent_list onto df 
    df['sentiment'] = sent_list
    df['sentiment_score'] = sent_score_list

    # create sentiment_df to have only id and sentiment
    sentiment_df = df[['id', 'sentiment_score', 'sentiment']]

    # append sentiment data into sentiment table
    for index, row in sentiment_df.iterrows():
        cur.execute(f"INSERT INTO sentiment (id, sentiment_score, sentiment) values ({row.id}, {row.sentiment_score}, '{row.sentiment}')")

    # update the chat_tracker with the sentiment data 
    update_sent_query = 'UPDATE chat_tracker_test c INNER JOIN sentiment s ON c.id = s.id SET c.sentiment = s.sentiment WHERE c.sentiment IS NULL;'
    update_score_query = 'UPDATE chat_tracker_test c INNER JOIN sentiment s ON c.id = s.id SET c.sentiment_score = s.sentiment_score WHERE c.sentiment_score is NULL;'
    reset_query = 'DELETE from sentiment'
    cur.execute(update_sent_query)
    cur.execute(update_score_query)
    cur.execute(reset_query)

    conn.commit()
    cur.close()



print(insertValues())
