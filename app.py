from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load a model from HuggingFace (you can change this to any model you prefer)
model = pipeline('text-generation', model='gpt2')  # You can change 'gpt2' to another model if needed

@app.route('/chat', methods=['POST'])
def chat():
    # Get the message from the POST request
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Generate a response using the model
    response = model(user_message, max_length=50)[0]['generated_text']

    # Return the response in JSON format
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
