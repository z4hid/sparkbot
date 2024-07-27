from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from src.model import NeuralNet
from src.utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cpu')

with open('./data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "models/model.pt"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SparkBot"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def handle_message():
    message = request.get_json()['message']
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                break
    else:
        response = "I do not understand..."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000,) # debug=True)