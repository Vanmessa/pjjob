from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sqlite3
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json


app = FastAPI()

class Data(BaseModel):
    UserID:str

class Data2(BaseModel):
    UserID:int
    IDDor:int
    rating:int

def loadModel():
    global predict_model
    predict_model = load_model('model.h5')
    

loadModel()

def updatedata():
    conn=db_connect()
    cursor=conn.cursor()
    


    return


def db_connect():
    con = sqlite3.connect('dor_rec.sqlite3')
    print("connect success")
    return con

async def predict(data):
    # connect sqllite
    con=db_connect()
    cur= con.cursor()

    # sqllite3
    combined = pd.read_sql_query("SELECT * FROM combined", con)
    combined_df = pd.DataFrame(data = combined)
    combined_df['DorRating'] = combined_df['DorRating'].values.astype(float)
    combined_df = combined_df.drop_duplicates(['UserID', 'Dormitory'])
    user_dor_matrix = combined_df.pivot(index='UserID', columns='Dormitory', values='DorRating')
    user_dor_matrix.fillna(0, inplace=True)
    users = user_dor_matrix.index.tolist()
    dors = user_dor_matrix.columns.tolist()
    user_dor_matrix = user_dor_matrix.to_numpy()

    # model
    filename = 'model.h5'
    predict_model = load_model(filename) 

    # sqllite3
    preds = predict_model(user_dor_matrix)
    preds = preds.numpy()
    pred_data = pd.DataFrame(preds)
    pred_data = pred_data.stack().reset_index()
    pred_data.columns = ['UserID', 'Dormitory', 'DorRating']
    pred_data['UserID'] = pred_data['UserID'].map(lambda value: users[value])
    pred_data['Dormitory'] = pred_data['Dormitory'].map(lambda value: dors[value])
    cur.execute("SELECT UserID, Dormitory FROM pred_data ORDER BY UserID, DorRating ASC")
    data1=cur.fetchall()

    # tolist
    d = {}
    for key, val in data1:
        d.setdefault(key, []).append(val)
    userid=data.UserID
    n = int(data.UserID)
    dor=d.get(n)

    return  userid,dor 


async def updaterating(data):  
        con=db_connect()
        cur= con.cursor()
        sqlite_insert_with_param = """INSERT INTO rating
                          (IDuser,IDDor,rating) 
                          VALUES (?, ?, ?);"""

        data_tuple = (data.UserID,data.IDDor,data.rating)
        cur.execute(sqlite_insert_with_param, data_tuple)
        con.commit()
        print("Python Variables inserted successfully into rating table")

        cur.close()

        
        return "success update"


async def updatetable():


    return

@app.post("/getclass/")
async def get_class(data: Data):
    # category, confidence = await predict(data)
    # res = {'class': category, 'confidence':confidence}

    userid,dor = await predict(data)
    res = { 'UserID':userid,'dormitory':dor}
    return {'results': res}



@app.post("/updateuser/")
async def updateuser(data: Data2):
    # category, confidence = await predict(data)
    # res = {'class': category, 'confidence':confidence}

    text2 = await updaterating(data)

    # await updatedata(data)
    return text2
