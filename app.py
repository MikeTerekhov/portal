import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json()
    func = data.get('function', 'linear')
    color = data.get('color', 'blue')
    
    x = range(0, 10)
    if func == 'linear':
        y = [v for v in x]
    elif func == 'quadratic':
        y = [v**2 for v in x]
    elif func == 'cubic':
        y = [v**3 for v in x]
    else:
        y = [0 for _ in x]
    
    plt.figure()
    plt.plot(x, y, color=color)
    plt.title(f'{func.capitalize()} Function')
    plt.xlabel('x')
    plt.ylabel('y')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return jsonify({'plot': img_data})

@app.route('/plot-csv', methods=['POST'])
def plot_csv():
    try:
        file = request.files.get('file')
        color = request.form.get('color', 'blue')
        
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Read the file based on extension
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file type. Use CSV or Excel files.'}), 400
        
        # Assume first two columns are x and y
        if len(df.columns) < 2:
            return jsonify({'error': 'File must have at least 2 columns (x and y)'}), 400
        
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        
        plt.figure()
        plt.plot(x, y, color=color, marker='o')
        plt.title('Data from Uploaded File')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({'plot': img_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
