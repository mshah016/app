from flask import Flask, render_template, request, jsonify 
import prediction as p

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("sent_model.html")


@app.route('/input', methods=['POST', 'GET'])
def input():
    data = request.get_json()
    print(data['content'])
    inputText = data['content']
    prediction = p.inputPrediction(inputText)
    print(prediction)
    obj = {
        "text": inputText,
        "sentiment": prediction,
        "score": 1.0
    }
    print(obj)
    return jsonify(obj)
   

if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='127.0.0.1:5000')


