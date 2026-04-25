# Smart Loan Approval System

An AI-powered web application that predicts loan approval probabilities using a **Random Forest Classifier**. Built with Python and Streamlit, this app evaluates applicant details, calculates risk scores, estimates EMIs, and provides transparent, rule-based explanations for its financial decisions.

# Live Demo
*[Insert your Streamlit Cloud app link here, e.g., https://your-app-url.streamlit.app/]*

# Features

* **Machine Learning Prediction:** Uses a Random Forest model trained on synthetic financial data to predict loan approval probability.
* **Risk & EMI Calculation:** Automatically calculates a 1-100 Risk Score and estimates monthly EMI based on a standard 10% interest rate over 2 years.
* **Business Logic Engine:** Combines ML probability with hard financial rules (e.g., Debt-to-Income ratio, minimum credit scores) to make final decisions.
* **Explainable AI:** Doesn't just give a "Yes" or "No". It provides specific reasons for conditional approvals or rejections (e.g., "EMI exceeds safe limit").
* **Interactive UI:** Clean, user-friendly interface built entirely in Streamlit.

# Tech Stack

* **Language:** Python
* **Frontend/Framework:** Streamlit
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Data Manipulation:** Pandas, NumPy

# How It Works

1. **Synthetic Data Generation:** The app generates 2,000 realistic applicant profiles with attributes for Income, Age, Loan Amount, and Credit Score.
2. **Model Training:** A `RandomForestClassifier` is trained on-the-fly using an 80/20 train-test split to recognize approval patterns.
3. **User Evaluation:** When a user enters their details, the model predicts the probability of approval, while the internal business engine checks for strict financial red flags.

