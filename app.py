from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=["https://miketerekhov.github.io"])

@app.route('/')
def home():
    return "Hello from Flask on Render!"

@app.route('/api/data')
def data():
    return jsonify({"msg": "This is data from Flask backend"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
