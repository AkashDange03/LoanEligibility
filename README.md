# Loan Prediction Web Application using ML
The Loan Prediction Web Application is a web-based tool that predicts loan approval based on user input. It utilizes a machine learning model to provide an estimated outcome for loan applications. This project aims to assist individuals and financial institutions in making informed decisions regarding loan approvals.

# Dataset used for model training
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

# Features
* User-friendly Interface: The web application offers a simple and intuitive user interface where users can easily input their information.

* Prediction: Once the user submits their details, the application processes the input and provides a prediction regarding loan approval.

* Model Integration: The application integrates a pre-trained machine learning model that has been trained on historical loan data.

# Technologies Used
* Node.js: The server-side runtime environment that powers the web application.
* Express.js: A fast and minimalist web framework for Node.js that handles routing and middleware.
* EJS: A templating engine that enables dynamic content rendering on the server.
* Python: The programming language used for machine learning model training and prediction.
* Pandas: A powerful data manipulation library in Python for handling structured data.
* scikit-learn: A popular machine learning library in Python that provides various algorithms and tools for model training and evaluation.
* Joblib: A Python library used for serializing and deserializing machine learning models.
* Git: Version control system for managing project codebase.

# Prerequisites
Before running the application, make sure you have the following installed:
* Node.js
* Python
* pip 
* npm

# Installation
1.Clone the repository:
```
git clone https://github.com/AkashDange03/LoanEligibility.git
```
2.Install the required Node.js dependencies:
```
cd loan-prediction-web-app
npm init
npm install
```

3.Install the required python dependencies:
```
pip install pandas joblib scikit-learn
```

# Usage
* Start the Node.js server:
```
node server.js
```
* Access the web application in your browser at http://localhost:3000.
* Fill out the loan prediction form with the required information.
* Click the "Submit" button.
* The application will process the input and display the prediction result.






