from flask import Flask,request, url_for, redirect, render_template
import json
import pickle
import numpy as np
import pyttsx3
from tensorflow import keras
import speech_recognition as sr
r = sr.Recognizer()
app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


with open("chatbot_data.json") as file:
    data = json.load(file)
@app.route('/chat',methods=['POST','GET'])
def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    while True:

        userInput=request.form['User_Text']
        engine = pyttsx3.init()
        if userInput.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([userInput]),
                                                                          truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])


        for i in data['intents']:
            if i['tag'] == tag:
                print( np.random.choice(i['responses']))
                engine.say(np.random.choice(i['responses']))
                engine.runAndWait()
                return render_template('index.html', bot=np.random.choice(i['responses']),userInput=userInput)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=False)