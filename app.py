import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load

pipeline = load("pipeline.pkl")
model = load("final_model.pkl")

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    array_data = [np.array(features)]
    prepared = pipeline.transform(array_data)
    prediction = model.predict(prepared)

    output = round(prediction[0]*100, 2)

    return render_template('index.html', prediction_text=f"You chance of admission into a top 30 graduate school program is {output}%")


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = pipeline.transform([np.array(list(data.value()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run()
