import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle

#wczytuje classe modelu flask
app = Flask(__name__)

#wczytuje nasz zapiklowany model i scaler
model = pickle.load(open('modele/model_final_xgb.pkl', 'rb'))
scaler = pickle.load(open('modele/scaler.pkl', 'rb'))

#pobiera nasz template strony
@app.route('/')
def home():
    return render_template('index.html')

#pobiera wprowadzone przez nas dane na stronie
@app.route('/predict',methods=['POST']) 
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(scaler.transform(final_features))

    output = ["poor, do not drink it. Life is too short" if prediction[0] == 1 else "good, i recommend"]

    return render_template('index.html', prediction_text="Your wine is  {}".format(output).replace("['", "").replace("']",""))

#tu chyba wymuszamy wpisanie danych
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)