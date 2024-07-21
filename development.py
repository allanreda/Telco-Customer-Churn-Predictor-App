import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#pd.reset_option('display.float_format')
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from joblib import dump


# Import raw data from Excel-file
raw_df = pd.read_csv("Telco-Customer-Churn.csv") 
raw_df.columns

###############################################################################
########################## Exploratory Data Analysis ##########################
###############################################################################

########################### Minor data preprocessing ##########################

# Drop CompanyID
raw_df.drop('customerID', axis=1, inplace=True)
#_______________________

# Replace empty string values with 0 for the TotalCharges-column
raw_df['TotalCharges'] = raw_df['TotalCharges'].replace(' ', 0)
# Convert TotalCharges-column to float
raw_df['TotalCharges'] = raw_df['TotalCharges'].astype(float)

#_______________________

# Convert SeniorCitizen-column from 0/1 to Yes/No
raw_df['SeniorCitizen'] = raw_df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

############################## Data Exploration ###############################

# Check balance of the target variable
raw_df['Churn'].value_counts()
raw_df['Churn'].value_counts(normalize=True) * 100

#____________________________ Categorical Columns _____________________________

# List of columns to create pie charts for
categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

# Loop to create pie charts for categorical columns
for column in categorical_columns:
    # Calculate value counts
    value_counts = raw_df[column].value_counts()
    
    # Plot pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, 
            colors=plt.cm.Paired.colors, textprops={'fontsize': 16})
    plt.title(f'Pie Chart for {column}', fontsize=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

#_____________________________ Continuous Columns _____________________________

# Check descriptive statistics for continuous columns
descriptive_stats = raw_df[['tenure','MonthlyCharges', 'TotalCharges']].describe()
print(descriptive_stats)

# List of columns relevant to plot
continuous_columns = ['tenure','MonthlyCharges', 'TotalCharges']

# Create histograms for each column
plt.figure(figsize=(15, 20))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(len(continuous_columns), 1, i)
    plt.hist(raw_df[column], bins=50, edgecolor='k')
    plt.title(f'Histogram of {column}', fontsize=20)
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

###############################################################################
##################### Feature Engineering / Preprocessing #####################
###############################################################################

#_________________________ Convert binary columns _____________________________

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
    
#_____________________ One-hot-encode multinomial columns _____________________

# Count number of each category in a column to determine which to exclude
raw_df['PaymentMethod'].value_counts()

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

# Convert all boolean columns to binary (1/0)
bool_columns = raw_df.select_dtypes(include='bool').columns
raw_df[bool_columns] = raw_df[bool_columns].astype(int)

###############################################################################
######################## Outliers and Multicollinearity #######################
###############################################################################

#_______________________ Check and handle outliers ____________________________

# Check descriptive statistics for continuous columns
descriptive_stats = raw_df[['tenure','MonthlyCharges', 'TotalCharges']].describe()
print(descriptive_stats)

# List of columns relevant to plot
continuous_columns = ['tenure','MonthlyCharges', 'TotalCharges']

# Create histograms for each column
plt.figure(figsize=(15, 20))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(len(continuous_columns), 1, i)
    plt.hist(raw_df[column], bins=50, edgecolor='k')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Impute outliers 
for column in ['tenure','MonthlyCharges', 'TotalCharges']:
    threshold = raw_df[column].quantile(0.90)
    raw_df.loc[raw_df[column]>threshold, column] = threshold

# Check descriptive statistics for continuous columns
descriptive_stats = raw_df[['tenure','MonthlyCharges', 'TotalCharges']].describe()
print(descriptive_stats)

#_____________________ Check and handle multicollinearity _____________________

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(raw_df.drop(columns=['Churn']).corr(method='pearson'),
            vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=14, weight='bold');

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

# Drop one of each correlated pairs
raw_df.drop(['TotalCharges', 'InternetService_Fiber optic'], axis=1, inplace=True)

###############################################################################
############################## Logistic Regression ############################
###############################################################################

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

#__________________________ Initial model creation ____________________________

# Initialize LogisticRegression model
model = LogisticRegression(max_iter=500,
                           class_weight='balanced') # Weight balancer for target variable

# Fit the model on X_train and y_train
model.fit(X_train, y_train)

# Predict on test data
y_preds = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_preds)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['retained', 'churned'])
disp.plot()
# Print classification report
print(classification_report(y_test, y_preds, target_names=['retained', 'churned']))


#_________________ Hyperparameter tuning with cross-validation ________________

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

# Get the best model
best_model = grid_search.best_estimator_

# Print best hyperparameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Perform cross-validation on the whole dataset with the best model
cv_scores = cross_val_score(best_model, X, y, cv=StratifiedKFold(n_splits=5), scoring='recall')

# Print cross-validation scores
print(f"Cross-Validation Recall Scores: {cv_scores}")
print(f"Average Cross-Validation Recall Score: {cv_scores.mean()}")

#___________________________ Final model evaluation ___________________________

# Predict on test data using the best model
final_y_preds = best_model.predict(X_test)

# Confusion matrix for final model
final_cm = confusion_matrix(y_test, final_y_preds)
# Plot the confusion matrix
final_disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=['retained', 'churned'])
final_disp.plot()
# Print classification report
print("Final Model Classification Report:")
print(classification_report(y_test, final_y_preds, target_names=['retained', 'churned']))

#___________________________ Intercept and coefficients _______________________

# Calculate odds
odds = np.exp(best_model.intercept_)
# Calculate probability of churn for when all variables are 0 
probability = odds / (1 + odds)
print(probability)

# Get coefficients for all the variables
model_coefficients = pd.DataFrame(best_model.coef_[0], index=X.columns)
model_coefficients = model_coefficients.sort_values(by=0, ascending=False)
print(model_coefficients)

# Plot coefficients
plt.figure(figsize=(10, 8))
model_coefficients[0].plot(kind='barh', color='blue')
plt.xlabel('Coefficient Value')
plt.ylabel('Predictor Variable')
plt.title('Logistic Regression Coefficients')

# Filter out and keep only the coefficients that are exactly zero
null_coefficients = model_coefficients.loc[model_coefficients[0] == 0]
print(null_coefficients)

# Save the model to a joblib file
dump(best_model, 'best_model.joblib')

# Save the scaler to a joblib file
dump(scaler, 'scaler.joblib')





