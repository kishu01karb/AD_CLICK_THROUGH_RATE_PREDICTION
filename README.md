Ad Click Prediction Model 📊
A machine learning project that predicts whether a user will click on an advertisement based on their demographic information and browsing behavior.
📋 Table of Contents

Overview
Features
Dataset
Model Performance
Installation
Usage
Project Structure
Technologies Used
Results
Future Improvements
Contributing
License

🎯 Overview
This project implements a Logistic Regression model to predict the Click-Through Rate (CTR) of online advertisements. The model analyzes user behavior patterns, demographics, and contextual information to determine the likelihood of ad engagement.
Key Question: How do we know which users will click our ads before showing them?
Solution: Train a machine learning model to predict clicks based on user behavior, ad type, and context.
✨ Features

Interactive Web Application built with Streamlit
High Accuracy: 82.15% prediction accuracy
Real-time Predictions based on user inputs
Visual Analytics: Confusion matrix and performance metrics
Scalable Architecture: Easy to retrain and deploy
User-Friendly Interface: Simple input forms with instant results

📊 Dataset
The model is trained on a dataset containing 10,000 records with the following features:
Input Features:

Daily Time Spent on Site (minutes): Time user spends on the website
Age: User's age
Area Income: Average income of the user's geographical area
Daily Internet Usage (minutes): Average daily internet consumption
Gender: Male/Female
Country: User's country (237 unique countries)

Target Variable:

Clicked on Ad: Binary (0 = No Click, 1 = Click)

Dataset Statistics:

Total Records: 10,000
Features: 10 columns
Missing Values: None
Class Distribution:

No Click: 50.83%
Click: 49.17%
✅ Well-balanced dataset



📈 Model Performance
Accuracy: 82.15%

Classification Report:
              precision    recall  f1-score   support
   No Click       0.83      0.82      0.82      1017
      Click       0.81      0.83      0.82       983
Confusion Matrix:

True Negatives: 830
False Positives: 187
False Negatives: 170
True Positives: 813

🚀 Installation
Prerequisites

Python 3.7 or higher
pip package manager

Step 1: Clone the Repository
bashgit clone https://github.com/yourusername/ad-click-prediction.git
cd ad-click-prediction
Step 2: Create Virtual Environment (Recommended)
bash# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
💻 Usage
Running the Streamlit App
bashstreamlit run app.py
The application will open in your default browser at http://localhost:8501
Using the Prediction Function in Python
pythonfrom model import predict_ad_click

# Example 1: High probability user
probability = predict_ad_click(
    Daily_Time_Spent_on_Site=30.2,
    Age=55,
    Area_Income=75000,
    Daily_Internet_Usage=120,
    Gender='Female',
    Country='United States'
)
print(f"Click Probability: {probability * 100:.2f}%")

# Example 2: Low probability user
probability = predict_ad_click(
    Daily_Time_Spent_on_Site=80,
    Age=22,
    Area_Income=44000,
    Daily_Internet_Usage=250,
    Gender='Male',
    Country='France'
)
print(f"Click Probability: {probability * 100:.2f}%")
Training the Model from Scratch
bashjupyter notebook Ad_Click_Prediction.ipynb
```

Run all cells to:
1. Load and explore the data
2. Perform feature engineering
3. Train the model
4. Evaluate performance
5. Save the trained model

## 📁 Project Structure
```
ad-click-prediction/
│
├── Ad_Click_Prediction.ipynb   # Main training notebook
├── app.py                        # Streamlit web application
├── ad_records.csv                # Dataset
├── ad_click_model.pkl            # Trained model
├── scaler.pkl                    # Feature scaler
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── assets/                       # Images and visualizations
    ├── confusion_matrix.png
    └── distribution.png
🛠 Technologies Used
Core Libraries:

pandas - Data manipulation and analysis
numpy - Numerical computing
scikit-learn - Machine learning algorithms
matplotlib - Data visualization
seaborn - Statistical visualizations
streamlit - Web application framework
joblib - Model serialization

Machine Learning:

Algorithm: Logistic Regression
Preprocessing: StandardScaler
Train-Test Split: 80-20
Cross-validation: Stratified sampling

📊 Results
Key Insights:

Time Spent on Site: Users who spend less time are more likely to click ads
Age Factor: Older users show higher engagement with advertisements
Income Correlation: Lower area income correlates with higher click rates
Internet Usage: Moderate internet users are most likely to click

Model Strengths:

✅ Balanced precision and recall
✅ Fast prediction time
✅ Interpretable results
✅ Scalable for production

Model Limitations:

Limited to features available in training data
Requires periodic retraining with fresh data
Country encoding requires known countries from training
