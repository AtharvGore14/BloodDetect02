from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from PIL import Image
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import torchvision.transforms as transforms
import torch.nn as nn
import threading  # To handle image prediction in separate threads

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to store prediction result
prediction_result = None

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, fullname TEXT, email TEXT UNIQUE, username TEXT UNIQUE, password TEXT)'
        )
        conn.commit()

init_db()

# Correct SimpleCNN architecture matching trained model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Model Loading
num_classes = 8  # Blood group classes
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('fingerprint_blood_group_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image Transformation
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Must match training normalization
])

# Predict function (synchronous, running in a separate thread)
def predict_image(file_path):
    global prediction_result  # Access the global variable
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))  # Ensure the image matches the model's input size
        img_tensor = data_transform(img).unsqueeze(0)

        blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O-', 'O+']
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction_result = blood_groups[predicted.item()]  # Store the result globally
    except Exception as e:
        prediction_result = f"Prediction failed: {str(e)}"

@app.route('/')
def home():
    return render_template('hospitalhomepage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            flash('Login successful!', 'success')
            return redirect(url_for('predict_blood_group'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
                         (fullname, email, username, hashed_password))
            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/predictor')
def predictor():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Start a thread for prediction
            thread = threading.Thread(target=predict_image, args=(file_path,))
            thread.start()
            thread.join()  # Wait for the thread to finish

            # Render the result page with the predicted blood group
            return render_template('result.html', blood_group=prediction_result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('predictor.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)  # Enable threaded mode
