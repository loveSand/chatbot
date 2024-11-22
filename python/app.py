import os
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained tokenizer and model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Flask app setup
app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = get_response(user_input)
    return jsonify({'response': response})

def get_response(user_input):
    # Encode input and get response from the model
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    # Generate responses from the model
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the response
    bot_output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output

print("template folder path:", os.path.join(app.root_path, 'templates'))

if __name__ == '__main__':
    app.run(debug=True)
