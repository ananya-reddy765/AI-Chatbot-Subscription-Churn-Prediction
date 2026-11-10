# AI Chatbot Subscription Churn Prediction

An intelligent machine learning application that predicts customer subscription churn by analyzing user engagement, subscription details, usage patterns, and customer experience.

The project uses a **custom Logistic Regression classifier built from scratch using NumPy** to estimate churn probability and classify customers into different risk levels through an interactive Flask web application.

## Overview

Customer churn occurs when users discontinue a subscription or stop using a service. Identifying customers who are likely to leave helps businesses understand engagement patterns and improve customer retention strategies.

This project provides a machine learning-based approach to predict subscription churn and visualize model performance through an interactive web interface.

## Key Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to stay or churn.
- **Churn Probability:** Displays the estimated probability of customer churn.
- **Risk Classification:** Categorizes customers into Low, Medium, and High Risk.
- **Customer Insights:** Uses engagement and subscription attributes to provide contextual explanations.
- **Analytics Dashboard:** Displays accuracy, precision, recall, F1-score, AUC, and confusion matrix.
- **Interactive Interface:** Provides dedicated Home, Prediction, Results, and Analytics pages.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Custom Logistic Regression |
| Numerical Computing | NumPy |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | Python |
| Model Storage | Pickle |
| Development Environment | VS Code |

## Machine Learning Approach

The project implements a Logistic Regression classifier from scratch using NumPy.

**Workflow:**

1. Data Collection – Load customer subscription and engagement data.
2. Exploratory Data Analysis – Analyze customer patterns and dataset characteristics.
3. Data Preprocessing – Encode categorical features and scale numerical values.
4. Model Training – Train a custom Logistic Regression classifier.
5. Model Evaluation – Evaluate performance using classification metrics.
6. Prediction – Generate churn probabilities and customer risk classifications.

### Input Features

- Usage Duration
- Subscription Plan
- Daily Activity
- Session Length
- Usage Frequency
- Experience Rating
- Last Active Status

### Model Performance

The application displays the following evaluation metrics:

| Metric | Score |
|---|---:|
| Accuracy | 84.5% |
| Precision | 67.8% |
| Recall | 76.9% |
| F1-Score | 72.1% |
| ROC-AUC | 88.6% |

*These are the evaluation values configured in the application's analytics dashboard.*

## Project Structure

```text
AI-Chatbot-Subscription-Churn-Prediction/
│
├── app/
│   ├── app.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       ├── home.html
│       ├── predict.html
│       ├── result.html
│       ├── analytics.html
│       └── sidebar.html
│
├── data/
│   ├── chatbot_churn.csv
│   ├── preprocess.py
│   ├── generate_data.py
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   ├── ordinal_maps.pkl
│   └── *.png
│
├── models/
│   ├── logistic_regression.py
│   ├── train.py
│   ├── evaluate.py
│   ├── trained_model.pkl
│   └── best_threshold.pkl
│
├── notebooks/
│   └── eda.py
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ananya-reddy765/AI-Chatbot-Subscription-Churn-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Chatbot-Subscription-Churn-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install flask numpy scikit-learn
```

### 6. Run the Application

```bash
python app/app.py
```

### 7. Open in Browser

Visit:

```text
http://127.0.0.1:5000
```

## Application Pages

- **Home:** Introduction to the customer churn prediction system.
- **Prediction:** Enter customer information to generate predictions.
- **Results:** View churn probability, risk level, and customer insights.
- **Analytics:** Explore model evaluation metrics and performance visualizations.

## Future Enhancements

- Integration with real-time subscription data.
- Advanced machine learning models for performance comparison.
- Personalized customer retention recommendations.
- Historical churn analysis and interactive visualizations.
- Deployment on a cloud platform.

## Learning Outcomes

- Implementing Logistic Regression from scratch using NumPy.
- Understanding data preprocessing and feature scaling.
- Building and integrating a machine learning model with Flask.
- Developing an interactive web application.
- Evaluating classification models using performance metrics.

---

**Developed as a Machine Learning project focused on customer behaviour analysis and subscription churn prediction.**
