import gradio as gr
from joblib import load
import sklearn
import pandas as pd
import os

###############################################################################
############################# Backend Functions ###############################
###############################################################################

# Load the model and scaler
model = load('best_model.joblib')
scaler = load('scaler.joblib')

# Function for adding one-hot-encoded variables to the input data
def add_OHE_variable(input_data, var_name, zeros = int, excluded_category = str, categories = list):
    # Check if the chosen category is the same as the excluded
    if var_name == excluded_category:
        # If excluded_category, all dummy variables are assigned 0
        input_data += [0] * zeros
    else: # else assign 1 (True) to the chosen category and 0 to the rest
        input_data += [int(var_name == category) for category in categories if category != excluded_category]

# Function to predict car prices
def predict(SeniorCitizen, Partner, Dependents, tenure, PhoneService, PaperlessBilling, MonthlyCharges,
            MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
            TechSupport, Contract, PaymentMethod):
    
    # Initialize input data list
    input_data = []
    
    #____________
    
    # Map binary variables to 0/1, and add one-hot-encoded variables with each 
    # of its excluded category also. It is important that the variables are 
    # added in the same order as was given to the initial model.
    
    # Scale the continuous variables
    continuous_data = scaler.transform([[tenure, MonthlyCharges]])
    
    # Input for excluded gender variable
    input_data.append(0)   
    
    binary_mapping = {'Yes': 1, 'No': 0}
    input_data.append(binary_mapping[SeniorCitizen]) 
    input_data.append(binary_mapping[Partner]) 
    input_data.append(binary_mapping[Dependents]) 
    
    input_data.append(continuous_data[0][0])
    
    input_data.append(binary_mapping[PhoneService]) 
    input_data.append(binary_mapping[PaperlessBilling]) 
    
    input_data.append(continuous_data[0][1])
    
    add_OHE_variable(input_data, MultipleLines, 2, 'No phone service', ['No', 'Yes', 'No phone service'])  
    add_OHE_variable(input_data, InternetService, 1, 'Fiber optic', ['DSL'])
    add_OHE_variable(input_data, OnlineSecurity, 2, 'No internet service', ['No', 'Yes', 'No internet service'])
    add_OHE_variable(input_data, OnlineBackup, 2, 'No internet service', ['No', 'Yes', 'No internet service'])
    add_OHE_variable(input_data, DeviceProtection, 2, 'No internet service', ['No', 'Yes', 'No internet service'])
    add_OHE_variable(input_data, TechSupport, 2, 'No internet service', ['No', 'Yes', 'No internet service'])
    # Input for excluded StreamingMovies variable
    input_data.extend([0,0])
    # Input for excluded StreamingTV variable
    input_data.extend([0,0])
    add_OHE_variable(input_data, Contract, 2, 'One year', ['Month-to-month', 'Two year', 'One year'])
    add_OHE_variable(input_data, PaymentMethod, 3, 'Credit card (automatic)', ['Bank transfer (automatic)', 'Electronic check', 'Mailed check', 'Credit card (automatic)'])
    #____________

    # Feature names in the same order as used during model training
    feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 
                     'MonthlyCharges', 'MultipleLines_No', 'MultipleLines_Yes', 'InternetService_DSL', 
                     'OnlineSecurity_No', 'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_Yes', 
                     'DeviceProtection_No', 'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_Yes', 
                     'StreamingMovies_No', 'StreamingMovies_Yes', 'StreamingTV_No', 'StreamingTV_Yes', 
                     'Contract_Month-to-month', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 
                     'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    # Convert to dataframe
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Use the model to predict
    prediction = model.predict(input_df)
    
    # Define return statements for predictions
    if prediction[0] == 1:
        return "This customer has a high risk of leaving. Do something about it"
    else:
        return "This customer is good for now. Keep business as usual"


###############################################################################
################################ Interface ####################################
###############################################################################

# CSS to center title
css = """
h1 {
    text-align: center;
    display: block;
}

"""
# Define Gradio block 
with gr.Blocks(theme='freddyaboulton/dracula_revamped', title = 'Telco Customer Churn Predictor', css = css) as app:
    # Set title
    gr.Markdown("# Telco Customer Churn Predictor")
    
    # Define row for input fields
    with gr.Row():
        # First column of input fields
        with gr.Column():
            tenure = gr.Number(label="Tenure") 
            MonthlyCharges = gr.Number(label="Monthly Charges") 
            SeniorCitizen = gr.Dropdown(choices=['Yes', 'No'], label="Senior Citizen")
            Partner = gr.Dropdown(choices=['Yes', 'No'], label="Partner")
            Dependents = gr.Dropdown(choices=['Yes', 'No'], label="Dependents")
        # Second column of input fields
        with gr.Column():
            PhoneService = gr.Dropdown(choices=['Yes', 'No'], label="Phone Service")
            PaperlessBilling = gr.Dropdown(choices=['Yes', 'No'], label="Paperless Billing")
            MultipleLines = gr.Dropdown(choices=['No', 'Yes', 'No phone service'], label="Multiple Lines")
            InternetService = gr.Dropdown(choices=['Fiber optic', 'DSL', 'No'], label="Internet Service")
            OnlineSecurity = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label="Online Security")
        # Third column of input fields
        with gr.Column():
            OnlineBackup = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label="Online Backup")
            DeviceProtection = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label="Device Protection")
            TechSupport = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label="Tech Support")
            Contract = gr.Dropdown(choices=['Month-to-month', 'Two year', 'One year'], label="Contract")
            PaymentMethod = gr.Dropdown(choices=['Bank transfer (automatic)', 'Electronic check', 'Mailed check', 'Credit card (automatic)'], label="Payment Method")
    # Define row for submit and output field
    with gr.Row():
        # Single column for submit and output field
        with gr.Column():
         submit_button = gr.Button("Submit")
         prediction = gr.Text(label="Prediction")
    # Define arguments for the submit button
    submit_button.click(
        fn=predict, # The 'predict' function
        inputs=[SeniorCitizen, Partner, Dependents, tenure, PhoneService, PaperlessBilling, 
                MonthlyCharges, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
                DeviceProtection, TechSupport, Contract, PaymentMethod], # Inputs
        outputs=prediction # Output
    )

# Launch app
app.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7861)))
