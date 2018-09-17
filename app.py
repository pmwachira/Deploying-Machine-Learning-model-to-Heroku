from flask import Flask,abort,jsonify,request,render_template

from sklearn.externals import joblib
import numpy as np
import json

gbr=joblib.load('car_price_predictor_notebook/model.pkl')

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def input_to_one_hot(data):
    #init target output as a zero array
    enc_input=np.zeros(62)
    enc_input[0] = data['year_model']
    enc_input[1] = data['mileage']
    enc_input[2] = data['fiscal_power']
    # get the array of marks categories
    marks = ['Peugeot', 'Renault', 'Citroen', 'Mercedes-Benz', 'Ford', 'Nissan',
             'Fiat', 'Skoda', 'Hyundai', 'Kia', 'Dacia', 'Opel', 'Volkswagen',
             'mini', 'Seat', 'Isuzu', 'Honda', 'Mitsubishi', 'Toyota', 'BMW',
             'Chevrolet', 'Audi', 'Suzuki', 'Ssangyong', 'lancia', 'Jaguar',
             'Volvo', 'Autres', 'BYD', 'Daihatsu', 'Land Rover', 'Jeep', 'Chery',
             'Alfa Romeo', 'Bentley', 'Daewoo', 'Hummer', 'Mazda', 'Chrysler',
             'Maserati', 'Cadillac', 'Dodge', 'Rover', 'Porsche', 'GMC',
             'Infiniti', 'Changhe', 'Geely', 'Zotye', 'UFO', 'Foton', 'Pontiac',
             'Acura', 'Lexus']
    cols = ['year_model', 'mileage', 'fiscal_power', 'fuel_type_Diesel',
            'fuel_type_Electrique', 'fuel_type_Essence', 'fuel_type_LPG',
            'mark_Acura', 'mark_Alfa Romeo', 'mark_Audi', 'mark_Autres', 'mark_BMW',
            'mark_BYD', 'mark_Bentley', 'mark_Cadillac', 'mark_Changhe',
            'mark_Chery', 'mark_Chevrolet', 'mark_Chrysler', 'mark_Citroen',
            'mark_Dacia', 'mark_Daewoo', 'mark_Daihatsu', 'mark_Dodge', 'mark_Fiat',
            'mark_Ford', 'mark_Foton', 'mark_GMC', 'mark_Geely', 'mark_Honda',
            'mark_Hummer', 'mark_Hyundai', 'mark_Infiniti', 'mark_Isuzu',
            'mark_Jaguar', 'mark_Jeep', 'mark_Kia', 'mark_Land Rover', 'mark_Lexus',
            'mark_Maserati', 'mark_Mazda', 'mark_Mercedes-Benz', 'mark_Mitsubishi',
            'mark_Nissan', 'mark_Opel', 'mark_Peugeot', 'mark_Pontiac',
            'mark_Porsche', 'mark_Renault', 'mark_Rover', 'mark_Seat', 'mark_Skoda',
            'mark_Ssangyong', 'mark_Suzuki', 'mark_Toyota', 'mark_UFO',
            'mark_Volkswagen', 'mark_Volvo', 'mark_Zotye', 'mark_lancia',
            'mark_mini']
    redefined_user_input='mark_'+data['mark']

    mark_column_index=cols.index(redefined_user_input)

    enc_input[mark_column_index]=1

    fuel_type = ['Diesel', 'Essence', 'Electrique', 'LPG']
    redefined_user_input='fuel_type_'+data['fuel_type']
    fuelType_column_index=cols.index(redefined_user_input)

    enc_input[fuelType_column_index]=1
    return enc_input

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