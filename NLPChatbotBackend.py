import os

from flask import Flask
from flask_jwt_extended import JWTManager

# NLP imports

# Initialize Flask app and JWT
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

jwt_manager = JWTManager(app)  # Use a different variable name for the JWTManager instance