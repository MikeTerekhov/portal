from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask on Render!"

@app.route('/api/data')
def data():
    return jsonify({"msg": "This is data from Flask backend"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
