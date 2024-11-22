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
app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Keep track of the conversation history
chat_history_ids = None

# Travel fallback responses
travel_fallbacks = {
    "recommend": "Some great travel destinations include Japan, Italy, and New Zealand!",
    "travel": "Traveling is all about exploring new places, experiencing cultures, and relaxing.",
    "places": "I suggest visiting Bali for beaches, Kyoto for tradition, or Iceland for nature!",
    "default": "I'm here to help with your travel questions. Can you ask in a different way?"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    user_input = request.json.get('message', '')
    response = get_response(user_input)
    return jsonify({'response': response})

def get_response(user_input):
    """
    Generate a response for the user's input, either using fallback logic or the DialoGPT model.
    """
    global chat_history_ids

    # Normalize input for fallback checks
    user_input_lower = user_input.lower()

    # Check for fallback responses
    if "recommend" in user_input_lower:
        return travel_fallbacks['recommend']
    elif "travel" in user_input_lower or "place" in user_input_lower:
        return travel_fallbacks['places']
    elif "what is travel" in user_input_lower:
        return travel_fallbacks['travel']

    # Tokenize the input
    try:
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

        # Append to conversation history if it exists
        if chat_history_ids is not None:
            input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)

        # Generate a response using the model
        chat_history_ids = model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )
        bot_output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Return fallback response if model response is incoherent
        if not bot_output.strip() or len(bot_output.split()) < 3:
            return travel_fallbacks['default']

        return bot_output

    except Exception as e:
        # Log errors and return a fallback response
        print(f"Error during response generation: {e}")
        return travel_fallbacks['default']

# Print the template folder path for debugging
print("Template folder path:", os.path.join(app.root_path, 'templates'))

if __name__ == '__main__':
    app.run(debug=True)
