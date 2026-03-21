# IMPORTS --------------------------

from flask import Flask, render_template, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Configurations (optional but standard)
app.config['SECRET_KEY'] = os.environ.get('commkey')
app.config['DEBUG'] = True

# Home route
@app.route('/')
def home():
    return "Hello, Flask is running!"

# Example route with GET and POST
@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        data = request.json
        return jsonify({"received": data}), 200
    else:
        return jsonify({"message": "Send a POST request"}), 200

# Example dynamic route
@app.route('/user/<name>')
def user(name):
    return f"Hello, {name}!"

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)