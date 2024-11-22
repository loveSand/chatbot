import os
from flask import Flask, request, jsonify, render_template

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
    if "hello" in user_input.lower():
        return "Hi there! How can I help with your travels?"
    elif "destination" in user_input.lower():
        return "I suggest visiting Bali, Rome, or Kyoto!"
    else:
        return "I'm here to assist with your travel questions"

print("template folder path:", os.path.join(app.root_path, 'templates'))

if __name__ == '__main__':
    app.run(debug=True)