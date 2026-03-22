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

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)