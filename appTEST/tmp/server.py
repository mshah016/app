from flask import Flask, render_template, request
import tf_globalAvg as tga

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("sent_model.html")

@app.route('/input', methods=['POST'])
def inputString():
    inputText = request.form['text']
    print(inputText)
    prediction = tga.inputPrediction(inputText)
    print(prediction)
    return render_template("sent_model.html", prediction = prediction, inputText = inputText)

if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='127.0.0.1:5000')