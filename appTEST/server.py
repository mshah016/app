from flask import Flask, render_template, request, jsonify 
#import tf_globalAvg as tga
import prediction as p

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("sent_model.html")


@app.route('/hello', methods=['GET', 'POST'])
def hello():

    # POST request
    if request.method == 'POST':
        print('Incoming..')
        print(request.get()) 
        return 'OK', 200

    # GET request
    else:
        print(request.get())
        inputText = request.get()
        message = {'text': inputText}
        return jsonify(message)  # serialize and use JSON headers

#@app.route('/input', methods=['GET'])
#def inputString():

#     #POST request
#    if request.method == 'POST':
#        print("Incoming...")
#        print(request)
#        #inputText = request
#        #print("User Input Text: ")
#        #print(inputText)
#    else:
#        print("this is a get request")
   

if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='127.0.0.1:5000')