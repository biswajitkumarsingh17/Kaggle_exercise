from flask import Flask, render_template, request
import requests
import flask
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('rfr (2).pkl', 'rb'))


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        downs = request.form['downs'].replace(" ", "")
        gilded = request.form['gilded'].replace(" ", "")
        controversiality = request.form['controversiality'].replace(" ", "")

        downs, gilded, controversiality = int(downs), int(gilded), int(controversiality)

        def rfr_predict(downs, gilded, controversiality):
            x = np.zeros(120)
            x[0] = downs
            x[1] = gilded
            x[2] = controversiality

            return model.predict([x])[0]

        result = rfr_predict(downs, gilded, controversiality)

        print(result)

        return render_template('results.html', result=result)
    else:
        return render_template('index.html')
        


if __name__ == "__main__":
    app.run(debug=True)

