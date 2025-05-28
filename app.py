from flask import Flask, render_template, request, redirect, session, url_for
from flask_mysqldb import MySQL
import MySQLdb.cursors
import bcrypt
import pickle
import numpy as np
import tenseal as ts
import math
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'sasika1234'
app.config['MYSQL_DB'] = 'thyroid_prediction'

mysql = MySQL(app)

# ====== Load Model and Required Objects ======
def load_model_and_context():
    print("Loading encryption context...")
    try:
        # Load encryption context with secret key
        with open('encryption_context.tenseal', 'rb') as f:
            context = ts.context_from(f.read())
        
        # Load full model data
        with open('full_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            # Convert decrypted weights back to encrypted form
            weights = model_data['weights']
            context_from_model = ts.context_from(model_data['encryption_context'])
            trained_weights = [ts.ckks_vector(context_from_model, w) for w in weights]
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']
            
        return context, trained_weights, scaler, label_encoder
    except Exception as e:
        print(f"Error loading model and context: {str(e)}")
        raise e

# Load all required components
context, trained_weights, scaler, label_encoder = load_model_and_context()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        mysql.connection.commit()
        cur.close()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')

        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
            session['username'] = username
            return redirect('/home')
        else:
            return "Invalid Credentials. Try Again."

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html')
    return redirect('/login')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect('/login')

    if request.method == 'POST':
        try:
            # Get form data
            gender = 1 if request.form['Gender'] == 'Male' else 0
            goitre = 1 if request.form['Goitre'] == 'Yes' else 0
            age = float(request.form['Age'])
            tsh = float(request.form['TSH'])
            t3 = float(request.form['T3'])
            t4 = float(request.form['T4'])
            tt4 = float(request.form['TT4'])
            fti = float(request.form['FTI'])

            # Prepare input data
            user_input = np.array([[age, tsh, t3, t4, tt4, fti, gender, goitre]])
            
            # Normalize input
            normalized_input = scaler.transform(user_input)[0]
            
            # Encrypt normalized input
            encrypted_input = ts.ckks_vector(context, normalized_input.tolist())
            
            # Print encrypted user input
            print("Encrypted user input:", encrypted_input)
            # Get predictions for each class
            predictions = []
            for weight in trained_weights:
                # Compute dot product and decrypt
                score = encrypted_input.dot(weight)

                # Print encrypted prediction score
                print("Encrypted prediction score:", score)
                decrypted_score = score.decrypt()[0]
                predictions.append(decrypted_score)
            # Print decrypted prediction scores
            print("Decrypted prediction scores:", predictions)
            # Apply softmax to get probabilities
            predictions = np.array(predictions)
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / exp_preds.sum()
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            
            # Map to condition
            conditions = {0: "Normal", 1: "Hyperthyroidism", 2: "Hypothyroidism"}
            result = conditions.get(predicted_class, "Unknown")
            
            # Store prediction in database
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO predictions (username, age, tsh, t3, t4, tt4, fti, goitre, result) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session['username'], age, tsh, t3, t4, tt4, fti, goitre, result))
            mysql.connection.commit()
            cur.close()

            return render_template('result.html', result=result)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return render_template('predict.html', error="An error occurred during prediction. Please try again.")

    return render_template('predict.html')

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect('/login')

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT age, tsh, t3, t4, tt4, fti, goitre, result, prediction_date FROM predictions WHERE username = %s", (session['username'],))
    history_data = cur.fetchall()
    cur.close()

    return render_template('history.html', history=history_data)

if __name__ == '__main__':
    app.run(debug=True)