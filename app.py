from flask import Flask, request, jsonify
from transformers import pipeline
app = Flask(__name__)
model = pipeline('text-generation', model='gpt2')
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = model(user_message, max_length=50)[0]['generated_text']

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
