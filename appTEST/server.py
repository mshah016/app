from flask import Flask, render_template, request, jsonify 

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("sent_model.html")


@app.route('/hello', methods=['GET'])
def hello():
    incoming_data = request.get_json()
    print(incoming_data)
    complete_message = "complete"
    return complete_message

   

if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='127.0.0.1:5000')