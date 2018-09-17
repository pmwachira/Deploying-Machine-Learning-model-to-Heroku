from flask import Flask,abort,jsonify,request,render_template

from sklearn.externals import joblib
import numpy as np
import json

gbr=joblib.load('model.pkl')

app=Flask(__name__)

@app.route('/')
def home():
    render_template('home.html')

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    year_model=result['year_model']
    mileage=result['mileage']
    mark=result['mark']
    fiscal_power=result['fiscal_power']
    fuel_type=result['fuel_type']

    user_input={'year_model':year_model,'mileage':mileage,'fiscal_power':fiscal_power,'fuel_type':fuel_type,'mark':mark}

    a=input_to_one_hot(user_input)
    # get the price prediction
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)
    # return a json value
    return json.dumps({'price': price_pred});


if __name__ == '__main__':
    app.run(port=8080, debug=True)