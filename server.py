import pickle
import xgboost as xgb

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

model = pickle.load(open("xgboost_default.pickle", "rb"))

features = ['latitude', 'longitude', 'property_type', 'accommodates', 'bathrooms',
       'bedrooms', 'beds', 'bed_type', 'minimum_nights', 'maximum_nights',
       'number_of_reviews', 'instant_bookable', 'is_business_travel_ready',
       'entire_home', 'hotel_room', 'private_room', 'shared_room', 'wifi',
       'tv', 'internet', 'kitchen', 'paid_parking_off_premises', 'essentials',
       'washer', 'patio_or_balcony', 'isWE', 'week_number']

@app.route('/', methods=['GET'])
def home():
    return "Hello World!", 200

# On définit une route spécifique pour obtenir le prix avec une méthode GET
@app.route('/price', methods=['GET'])
def get_price():
    data = np.asarray(request.json['data']) # Format vecteur aplati
    data = data.reshape((-1, len(features)))  # On découpe par nombre de colonnes
    if not data is None:
        X = pd.DataFrame(data=data, columns=features)
        prices = model.predict(X)
        return jsonify({'prices': prices.flatten().tolist()}), 200
    return "An internal error happens !", 500

# Lorsque l'on exécute un script, cette portion est automatiquement exécutée par l'interpréteur.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
