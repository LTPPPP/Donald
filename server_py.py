# server_py.py 
import sys
import os

# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from flask_cors import CORS
import nltk
from routes.chatbot import chat_bp
from routes.auth import auth_bp
from routes.templates import templates_bp

nltk.download('punkt')

app = Flask(__name__, template_folder='./templates', static_folder='./static')
CORS(app)

app.register_blueprint(chat_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(templates_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
