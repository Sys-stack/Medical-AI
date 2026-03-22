# IMPORTS --------------------------

from flask import Flask, render_template, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Configurations (optional but standard)
app.config['SECRET_KEY'] = os.environ.get('commkey')
app.config['DEBUG'] = True

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("homepage.html")


# Example about page
@app.route('/about')
def about():

    return render_template("about.html")

#receive message from front end
@app.route("/chat", methods=["POST"])
def chat():
    """Receives { message }, processes it, returns { response }."""
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    return jsonify({"response": "fill"})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)