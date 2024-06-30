# Churn Prediction App

This repository contains a churn prediction app built with Streamlit. The app predicts whether a customer is likely to churn based on their attributes using a machine learning model.

## Project Description

Customer churn is a critical issue for businesses, especially in subscription-based services like telecommunications. Identifying customers who are likely to churn can help businesses take proactive measures to retain them. This project aims to build a user-friendly web application that predicts customer churn based on various customer attributes.

The app allows users to input customer data, such as tenure, monthly charges, and service preferences, and provides a prediction on whether the customer is likely to churn. The prediction is based on a pre-trained machine learning model.

### Key Features

- **Interactive User Interface**: The app provides an easy-to-use interface for entering customer details and obtaining churn predictions.
- **Machine Learning Model**: A pre-trained machine learning model is used to make accurate predictions.
- **Churn Probability**: The app not only predicts churn but also provides the probability of churn, giving more insight into the prediction.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ChurnPredictionApp.git
   cd ChurnPredictionApp
2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3.  Install the required packages:
  
    ```bash
      pip install -r requirements.txt
4.  Place your pre-trained model (model.pkl) and dataset (first_telc.csv) in the repository directory.


## Usage

1. Run the Streamlit app:
    
    ```bash
    streamlit run app1.py
2. Open your web browser and navigate to http://localhost:8501 to use the app.

## Streamlit App

You can interact with the model and visualize the results using the Streamlit app deployed [here](https://share.streamlit.io/app/telechurn8/).


## File Structure

- **app1.py**: The main Streamlit app file.
- **first_telc.csv**: The dataset used for the app.
- **model.pkl**: The pre-trained model file.
- **requirements.txt**: List of dependencies required to run the app.
- **Analysis.ipynb**: Jupyter notebook with data analysis.
- **ModelBuilding.ipynb**: Jupyter notebook with model building steps.

## Input Fields

The app requires the following inputs to predict churn:

- **SeniorCitizen**: Whether the customer is a senior citizen (0 or 1)
- **MonthlyCharges**: The monthly charges for the customer
- **TotalCharges**: The total charges for the customer
- **Gender**: The gender of the customer (Male or Female)
- **Partner**: Whether the customer has a partner (Yes or No)
- **Dependents**: Whether the customer has dependents (Yes or No)
- **PhoneService**: Whether the customer has phone service (Yes or No)
- **MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service)
- **InternetService**: The internet service of the customer (No, DSL, Fiber optic)
- **OnlineSecurity**: Whether the customer has online security (Yes, No, No internet service)
- **OnlineBackup**: Whether the customer has online backup (Yes, No, No internet service)
- **DeviceProtection**: Whether the customer has device protection (Yes, No, No internet service)
- **TechSupport**: Whether the customer has tech support (Yes, No, No internet service)
- **StreamingTV**: Whether the customer has streaming TV (Yes, No, No internet service)
- **StreamingMovies**: Whether the customer has streaming movies (Yes, No, No internet service)
- **Contract**: The contract type of the customer (One year, Month-to-month, Two year)
- **PaperlessBilling**: Whether the customer has paperless billing (Yes or No)
- **PaymentMethod**: The payment method of the customer (Credit card (automatic), Mailed check, Electronic check, Bank transfer (automatic))
- **Tenure**: The number of months the customer has been with the company

## Output

The app provides the following output based on the prediction:

- Whether the customer is likely to churn
- The probability of churn and not churn

## Acknowledgements

- Streamlit for providing an easy-to-use web framework for data apps
- scikit-learn for machine learning tools
- pandas and numpy for data manipulation and analysis

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
