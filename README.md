# Telco Customer Churn App
Try it out live at: https://telco-customer-churn-app-dot-sylvan-mode-413619.nw.r.appspot.com/

![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/b4b6d2f6-8d28-4f64-923f-fe4c0883a2b6)


## Overview  
A web-based ML-application, built on top of the popular "Telco Customer Churn"-dataset. The goal of this project was to train a logistic regression model to such a satisfactory performance, that its predictions could be used to provide actionable insights for the business. Furthermore, the goal was that the model should be usable by every employee, and not be reserved only for those with programming/data science skills - and that requires the creation of a UI in the form of a live web-application. The final idea being that employees should be able to input existing customer data into the UI, and recieve an output message on whether the particular customer had a high risk of leaving or not.   

This repository covers the following points:  
-Exploratory data analysis  
-Feature engineering and preprocessing  
-Outliers and multicollinearity  
-Logistic Regression training and hyperparameter tuning  
-Backend and frontend creation of Gradio web-application  
-GCP App Engine deployment with YAML-file  

## Explanation
### development.py  
In this section, we will cover the following:  
-Exploratory data analysis  
-Feature engineering and preprocessing  
-Outliers and multicollinearity  
-Logistic Regression training and hyperparameter tuning

#### Exploratory data analysis 
To get a sense of the data that I was working with, I started out with a brief exploratory analysis.  
The most important part to check in an exploratory data analysis, prior to any classification model, is the balance of the categories in the target variable.  
For this dataset, the target variable which is the churn column, the balance is 73,5% retained and 26,5% churned.  
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/c741e03b-b9de-4070-a11e-b0fd00413f95)  

While that is a large imbalance, it wasn't enough to cause considerations for sampling, later on. I therefore decided to proceed with the target variable as it was.

All 16 categorical variables were visualized with simple pie charts, and the results can be seen below.
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/4c7132de-8b3d-4897-a526-33222c598943)

The three continuous columns 'tenure', 'MonthlyCharges', and 'TotalCharges' were first checked using their descriptive statistics. 

```python
# Check descriptive statistics for continuous columns
descriptive_stats = raw_df[['tenure','MonthlyCharges', 'TotalCharges']].describe()
print(descriptive_stats)
```
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/24edabdf-ca2f-4bec-9514-e6b1307ef858)  
All three columns seemed to have a good varied distribution, which could be seen by the values of the quartiles relative to the mean value and the standard deviation. 
However, the maximum values were all significantly higher than the 75% quartile which indicated outliers, that needed to be handled.  
When visualized, the presence of outliers in the maximum values became further evident.
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/0d9866ed-af63-453f-8219-da3ed2457fdf)

#### Feature engineering and preprocessing
The data in this dataset mostly consists of categorical variables - either binary or multinomial. At this step of the process, they all had a string datatype and therefore needed to be encoded to integers to be processed by the logistic regression model. The continuous variables should also be standardized as a preprocessing step, but this is done later, to avoid interfering with the coming check for multicollinearity.

I encoded all the binary variables first, and did them all at once in a for loop, after defining which category should be encoded as 1.  
```python
# Function to perform binary encoding on columns
def convert_column_to_binary(raw_df, column, one_value):
    raw_df[column] = (raw_df[column] == one_value).astype(int)

# Dict with all binary columns and the corresponding values that should be encoded as 1
binary_columns = {'gender': 'Male',
                  'Partner': 'Yes',
                  'Dependents': 'Yes',
                  'PhoneService': 'Yes',
                  'PaperlessBilling': 'Yes',
                  'Churn': 'Yes',
                  'SeniorCitizen': 'Yes'}

# Loop through binary columns
for column, one_value in binary_columns.items():
    convert_column_to_binary(raw_df, column, one_value)
```

With the multinomial variables, I had to one-hot-encode them, in order to create multiple separate binary variables (dummy variables) for each category. To avoid multicollinearity, I defined a category for each variable that should be removed. Like I did with the binary variables, I encoded all the multinomial variables at once using a for loop. The loop used the function get_dummy_variables which did the one-hot-encoding and the removal of a chosen dummy variable.  
```python
# Function to one-hot-encode columns and create dummy variables
def get_dummy_variables(raw_df, column, drop_value):
    # One-hot-encode the column
    raw_df = pd.get_dummies(raw_df, columns = [column])
    # Remove one dummy variable to avoid multicollinearity
    raw_df = raw_df.drop([drop_value], axis=1)
    
    return raw_df

# Dict with all multinomial columns and the category which should be removed
dummy_columns = {'MultipleLines': 'MultipleLines_No phone service',
                 'InternetService': 'InternetService_No',
                 'OnlineSecurity': 'OnlineSecurity_No internet service',
                 'OnlineBackup': 'OnlineBackup_No internet service',
                 'DeviceProtection': 'DeviceProtection_No internet service',
                 'TechSupport': 'TechSupport_No internet service',
                 'StreamingMovies': 'StreamingMovies_No internet service',
                 'StreamingTV': 'StreamingTV_No internet service',
                 'Contract': 'Contract_One year',
                 'PaymentMethod': 'PaymentMethod_Credit card (automatic)'}

# Loop through multinomial columns to apply one-hot encoding
for column, drop_value in dummy_columns.items():
    raw_df = get_dummy_variables(raw_df, column, drop_value)
```

Finally, I converted all the dummy variables from booleans to integers.  
```python
# Convert all boolean columns to binary (1/0)
bool_columns = raw_df.select_dtypes(include='bool').columns
raw_df[bool_columns] = raw_df[bool_columns].astype(int)
```

#### Outliers and multicollinearity  

In the exploratory data analysis, it became evident that there were outliers in all three of the continuous columns that needed to be handled.
I didn't want to remove the extreme values entirely, because I wanted to maintain the integrity of the data distribution, but at the same time I also needed to ensure a robust and reliable model which performance wouldn't be affected by extreme outliers. I therefore decided that a good solution would be to cap the extreme values by replacing values above the 90th percentile with the values of the 90th percentile.  
```python
# Impute outliers 
for column in ['tenure','MonthlyCharges', 'TotalCharges']:
    threshold = raw_df[column].quantile(0.90)
    raw_df.loc[raw_df[column]>threshold, column] = threshold
```
The descriptive statistics for the continous columns looked like this afterwards. It could now be confirmed that the extreme outliers had been handled gracefully, while still maintaining the integrity of the data distribution.  
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/4fd2d1c5-cde4-4fcd-b9bf-bc36e1839673)

I could then continue to check for multicollinearity between all the predictor variables. I started with plotting a correlation matrix.  
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/509b7464-7ced-4704-acee-19bf746419f2)  
Overall it didn't look like the dataset was riddled with multicollinearity. I decided to pinpoint and eliminate all variable pairs with a correlation greater than 0.7. The choice of setting the threshold to 0.7 was arbitrary and can be changed higher or lower, due to your own preferences.  
```python
# Calculate the correlation matrix
correlation_matrix = raw_df.drop(columns=['Churn']).corr(method='pearson')

# Create a mask to get only the upper triangle of the matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Apply the mask to the correlation matrix
upper_triangle = correlation_matrix.where(mask)

# Find all pairs with correlation greater than 0.7
high_correlation_pairs = []
for column in upper_triangle.columns:
    for row in upper_triangle.index:
        if upper_triangle.loc[row, column] > 0.7 and row != column:
            high_correlation_pairs.append((row, column, upper_triangle.loc[row, column]))

# Print highly correlated pairs
print(high_correlation_pairs)
```
Luckily, there were only two pairs with a correlation greater than 0.7.  
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/a8ea92f5-afd9-4ae3-8acb-6ecf1f54ac64)  
So I eliminated one of each variable pair and thereby ensured parameter stability for the model I was about to train.
```python
# Drop one of each correlated pairs
raw_df.drop(['TotalCharges', 'InternetService_Fiber optic'], axis=1, inplace=True)
```

#### Logistic Regression training and hyperparameter tuning

The data was now ready for the actual model building.  
I proceeded to separate the predictor variables from the target variables and standardized the two continuous variables. The standardization was done to ensure uniformity in the data range, so that both variables contribute equally to the model.  
```python
# Copy dataframe
ml_df = raw_df.copy()

# Separate features and target
X = ml_df.drop('Churn', axis=1)
y = ml_df['Churn']

# Identify continuous columns
continuous_cols = ['tenure', 'MonthlyCharges']
# Standardize the continuous variables
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
```

The data was then split into training and test sets using a 75/25 ratio, while maintaining the same distribution of the target variable in both sets by using the "stratify" argument.  
```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
```

Initially, the first logistic regression model was fitted on the training sets without any specified hyperparameters. I set the "class_wheight" argument to "balanced", to combat the class imbalance of the target variable, which was discovered in the EDA. This argument should ensure that the model pays more attention to the minority class (churned), and treats it equally as the majority class (retained).  
```python
# Initialize LogisticRegression model
model = LogisticRegression(max_iter=500,
                           class_weight='balanced') # Weight balancer for target variable

# Fit the model on X_train and y_train
model.fit(X_train, y_train)
```

The confusion matrix for that model clearly showed that it was a lot better at predicting the retained customers than the churned customers.  
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/2a395f82-dffc-4c41-aa25-0aa9617396dc)  

To get a more precise understanding, I also looked at the classification report.  
![image](https://github.com/allanreda/Telco-Customer-Churn-App/assets/89948110/d8b6f194-aa79-4132-aeb1-4bdff0385053)  
It became evident that the model performed strongly in identifying the retained customers, while performing more moderately in identifying the churned customers. The precision score for the "churned" class was 0.52 which effectively means that 48% of the customers that were predicted as churned, were actually retained. That in itself is a relatively poor performance. The recall score for the same class, on the other hand, is significantly higher, at 0.79. This means that the model was able to correctly identify 79% of the customers that were churned.  

For this specific use case, that could be relatively ideal. In practice, it means that the model is effective at identifying customers who are likely to churn, while also classifying customers who are not at risk of churning, as likely to churn. Depending on the company's situation and their available resources, they would be able to identify and take proactive measures (e.g. better service or exclusive deals) for most customers likely to churn while also improving customer relations by engaging with customers who are not at immediate risk.  

Based on this, I decided to proceed with the model and tried to improve it, by tuning its hyperparameters.  
```python
# Define the hyperparameter grid
param_grid = [
    {
        'C': [0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 10, 20, 30, 50, 80, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],  # Regularization type
        'solver': ['liblinear']  # Solver that supports L1 and L2 regularization
    },
    {
        'C': [0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 10, 20, 30, 50, 80, 100],  # Regularization strength
        'penalty': ['l2'],  # Regularization type
        'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']  # Solvers that support only L2 regularization
    },
    {
        'C': [0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 10, 20, 30, 50, 80, 100],  # Regularization strength
        'penalty': ['elasticnet'],  # Regularization type
        'solver': ['saga'],  # Solver that supports ElasticNet regularization
        'l1_ratio': [0, 0.5, 1]  # ElasticNet mixing parameter
    },
    {
        'C': [0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 10, 20, 30, 50, 80, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],  # Regularization type
        'solver': ['saga']  # Solver that supports L1 and L2 regularization
    }
]

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=2000, class_weight='balanced')

# Set up GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5), scoring='recall', n_jobs=-1)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)
```

First, I defined the hyperparameter grid with multiple variations of settings to try. The grid contains four separate entries to accommodate the compatibility constraints between the 'penalty' and 'solver' parameters. Each solver supports different types of penalties, so they needed to be grouped accordingly.  

The "GridSearchCV" function was then set up to perform a grid search with a five-fold cross-validation, optimizing for the recall score. The recall score is used, because the function will then, by default, optimize for the recall of the positive class, (churned), which is what I deemed most relevant in this case. The cv=StratifiedKFold(n_splits=5) ensures that each of the five folds has the same proportion of classes, and n_jobs=-1 allows the use of all available CPU cores to speed up the computation.  

The average recall score for the five folds turned out to be 0.79 - the same as the recall score from the initial model.  
![image](https://github.com/user-attachments/assets/4df9e56a-ded2-4bb4-b6e0-d5a3e4a8e082)  
The classification report was also identical to the one from the initial model.  
![image](https://github.com/user-attachments/assets/2c3e8e93-471e-4309-a014-d0a59d9a1294)  
Therefore, I concluded that the model wasn't able to be further improved by the hyperparameter tuning, and proceeded with the model I had.  

As a final step, before I started building the app, I decided to take a look at the intercept and coefficients, to get a better understanding of the final model.  
```python
# Calculate odds
odds = np.exp(best_model.intercept_)
# Calculate probability of churn for when all variables are 0 
probability = odds / (1 + odds)
print(probability)

# Get coefficients for all the variables
model_coefficients = pd.DataFrame(best_model.coef_[0], index=X.columns)
model_coefficients = model_coefficients.sort_values(by=0, ascending=False)
print(model_coefficients)
```

When doing a logistic regression model, the intercept represents the log-odds of the outcome, if all predictor variables are zero. When the log-odds is exponentiated, we get the odds which can then be used in the logistic function to calculate a probability, to better interpret the model.  
![image](https://github.com/user-attachments/assets/33720a8a-4965-471d-acdb-42f47186dcbb)  
In this case, I got a probability of 0.48022312. In practice, that means that there is a 48% chance of churn, when all predictor variables are zero. One could say that 48% is the starting point, and knowing that will therefore provide a better insight into how much the coefficients are affecting that starting point.  

From the plot of the coefficients of the predictor variables, it became apparent that some variables had significantly more influence on the model than others, and that some variables had no influence at all.  
![image](https://github.com/user-attachments/assets/ecc28a45-d738-4db6-8c15-486647cc09e8)  
For example, 'MonthlyCharges' has the highest positive coefficient, indicating that an increase in monthly charges significantly raises the odds of a customer churning. 'Contract_Month-to-month' also shows a strong positive influence, meaning customers with a month-to-month contract are more likely to churn compared to those with longer-term contracts. 'tenure' and 'PhoneService' have the largest negative coefficients, indicating that longer tenure and having phone service are associated with lower churn rates. All this information could be useful for the company if they want to take action on the model predictions, such as creating customer retention strategies, as they now know where they should put more focus.  

For the predictor variables that had no influence on the model (coefficient of exactly zero), I decided I had to pinpoint and remove them, to improve the usability of the application I was about to build. I saw no reason as to why users should input data that had no impact on the outcome of the prediction.  
![image](https://github.com/user-attachments/assets/3cc31781-e073-445c-93a8-27860a9f90f3)  
As can be seen on the screenshot, the variables 'DeviceProtection_No', 'PaymentMethod_Mailed check', and 'MultipleLines_Yes' are dummies that are derived from variables whose other categories do have an impact on the model. For the sake of usability and explainability of the application and the model, I decided to keep those. For the rest of the variables, 'gender', 'StreamingTV_Yes', 'StreamingTV_No', 'StreamingMovies_Yes', and 'StreamingMovies_No' I decided that they were going to be excluded from the application.


### deployment/app.py  

#### Backend and frontend creation of Gradio web-application  

I decided to use the Gradio framework, because of its ease of use, both regarding building and usability. The app is the final product that the company can use to input customer data, and get a recommendation on what they should do, based on the app's churn prediction. The file contains two main parts, the backend and the frontend, which together makes the application.  

In the backend, I started by importing the logistic regression model, I had built earlier, and the scaler that was used for the continuous columns.  
```python
# Load the model and scaler
model = load('best_model.joblib')
scaler = load('scaler.joblib')
```

Also, for the readability of the backend prediction function, I created a function whose purpose was to take the input data from the multiple-choice fields that represent the one-hot-encoded variables and assign numerical values (0 or 1) to the model.
```python
# Function for adding one-hot-encoded variables to the input data
def add_OHE_variable(input_data, var_name, zeros = int, excluded_category = str, categories = list):
    # Check if the chosen category is the same as the excluded
    if var_name == excluded_category:
        # If excluded_category, all dummy variables are assigned 0
        input_data += [0] * zeros
    else: # else assign 1 (True) to the chosen category and 0 to the rest
        input_data += [int(var_name == category) for category in categories if category != excluded_category]
```
Here is a usage example:  
```python
    add_OHE_variable(input_data, MultipleLines, 2, 'No phone service', ['No', 'Yes', 'No phone service'])  
```

The full backend prediction function can be viewed in the file, but I will go over the main parts here.  
First and foremost, it is essential for the model that the variables are added in the same order as they were given to the model in the first place. The model itself takes only numerical values and has no chance of "knowing" which number belongs to which variable.  

The scaler that was imported at the beginning of the script, is used on the continuous variables. It is essential that it is the same scaler that was used on the training data, to keep the data consistent and in the same format that the model was initially trained on.  
```python
# Scale the continuous variables
    continuous_data = scaler.transform([[tenure, MonthlyCharges]])
```

For the variables that I decided to exclude earlier, due to their irrelevance to the model (coefficient of zero), I still needed to add some values to the model input, as placeholders to keep the variable order, since the model was trained with that specific order, and therefore expected it. This was done easily like in this example:    
```python
# Input for excluded StreamingMovies variable
    input_data.extend([0,0])
```

The final steps for the backend prediction function, was to convert the list of numerical data to a Pandas dataframe and feed it to the model to make a prediction. Finally, I defined two return statements, one for each of the possible outcomes (churn/retain).  
```python
# Convert to dataframe
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Use the model to predict
    prediction = model.predict(input_df)
    
    # Define return statements for predictions
    if prediction[0] == 1:
        return "This customer has a high risk of leaving. Do something about it"
    else:
        return "This customer is good for now. Keep business as usual"
```  

When building the interface I utilized Gradios Row() and Column() classes to divide the input, submit and output fields. There were 15 variables left that needed to be included in the app as input fields, so I made a Row() with three Column()s with five input fields each.  
For the submit button and output field, I made a second Row() with a single Column() containing both of these.  
```python
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
```

The last line of the script is responsible for starting the app. The 'server_name' parameter being set to '0.0.0.0' makes the app accessible on any network interface of the machine it’s running on. The 'int(os.environ.get('PORT', 7861))' part tries to get the port number from the environment variable PORT. If the PORT environment variable is not set, it defaults to 7861.  
```python
# Launch app
app.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7861)))
```  

### deployment/app.yaml

For the app's deployment, I chose Google Clouds App Engine. The files were uploaded through the Cloud SDK by navigating the local folder path and then using the "gcloud app deploy" command.

```shell
C:\Users\allan\AppData\Local\Google\Cloud SDK>cd "C:\Users\allan\Desktop\Personlige projekter\Telco Customer Churn App\Telco-Customer-Churn-App"

C:\Users\allan\Desktop\Personlige projekter\Telco Customer Churn App\Telco-Customer-Churn-App>ls
Telco-Customer-Churn.csv  deployment  development.py

C:\Users\allan\Desktop\Personlige projekter\Telco Customer Churn App\Telco-Customer-Churn-App>cd "deployment"

C:\Users\allan\Desktop\Personlige projekter\Telco Customer Churn App\Telco-Customer-Churn-App\deployment>ls
app.py  app.yaml  best_model.joblib  requirements.txt  scaler.joblib

C:\Users\allan\Desktop\Personlige projekter\Telco Customer Churn App\Telco-Customer-Churn-App\deployment>gcloud app deploy
```

The app.yaml file's purpose is to configure the application's deployment by defining its settings and parameters, including the runtime environment, entry point, and scaling. Without it, the application wouldn't be deployable.  
```yaml
service: telco-customer-churn-app # Name of the app
runtime: python312 
instance_class: F4

entrypoint: uvicorn app:app --host 0.0.0.0 --port $PORT 

handlers:
- url: /.*
  script: auto  # Automatically route requests to the application
  secure: always  # Force HTTPS

automatic_scaling:
  min_instances: 0 # Minimum number of instances
  max_instances: 1 # Maximum number of instances
  # No more than 5 requests at the same time
  max_concurrent_requests: 5
```  

A key point, that might need extra explanation, is my choice of scaling settings. I set the minimum number of instances to zero, which in practice means that no instances will be running, while the app is not being used. The advantage of that, is that it reduces the running costs of this application dramatically, which is desired since it was only deployed to showcase this project. The disadvantage of that decision is that the application is generally slow to start up, since it has to spin up a new instance each time. The number of maximum instances being set to one is solely because I don't expect much traffic, nor am willing to pay for it.

## Technologies  
-Scikit-Learn for machine learning and model evaluation  
-Gradio for interface building  
-App Engine (Google Cloud) and YAML for deployment  
-Pandas and NumPy for data manipulation  
-Matplotlib and Seaborn for data visualization  
