# BloodDetect 🩸

A revolutionary AI-powered system that predicts blood groups from fingerprint images using deep learning.

## ✨ Features

- **CNN-Based Prediction**: Utilizes Convolutional Neural Networks for accurate blood group classification
- **Fingerprint Analysis**: Predicts blood type from fingerprint patterns
- **Secure Processing**: Encrypted data handling for privacy protection
- **Web Interface**: User-friendly Flask web application
- **Fast Results**: Get predictions in seconds

## 🚀 Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite (optional)
- **Deployment**: (Specify if deployed, e.g., Heroku, AWS)

## 📂 Project Structure
BloodDetect/
├── static/ # Static files (CSS, JS, images)
│ ├── css/
│ ├── js/
│ └── images/
├── templates/ # HTML templates
├── models/ # Trained ML models
├── app.py # Flask application
├── requirements.txt # Python dependencies
└── README.md


## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AtharvGore14/BloodDetect.git
   cd BloodDetect
   python -m venv venv
source venv/bin/activate  
# On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python app.py
http://localhost:5000

📊 Dataset
We used a custom dataset of fingerprint images labeled with blood groups. The dataset contains:

5,000+ fingerprint samples

Balanced distribution across blood types (A, B, AB, O)

Both positive and negative Rh factors

🧠 Model Architecture
Our CNN model architecture:
Input Layer → Conv2D → MaxPooling → Dropout → 
Conv2D → MaxPooling → Dropout → 
Flatten → Dense → Output Layer

Achieved 92.4% accuracy on test data.

👥 Team Members
Atharv Gore (Developer)

Mayank Goplani (Developer)

Dnyanraj Gore (Developer)

Shraddha Golhar (Developer)

Anushka Gore (Developer)

Gaurav Gore (Developer)

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Our faculty guides at Vishwakarma Institute of Technology
Open-source contributors to TensorFlow and Flask
Research papers on fingerprint-blood group correlations

🌟 Future Scope
-Mobile application development

-Integration with hospital management systems

-Real-time prediction API

-Expanded blood parameter predictions
