import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

file_path = "synthetic_flight_passenger_data.csv"
data = pd.read_csv(file_path)

size= data.shape
print("Number of rows:", size[0])
print("Number of columns:", size[1])

print(data.columns)

# Strategy Target: Flight_Satisfaction_Score 
# Selected Features: Airline Departure_Airport Arrival_Airport Departure_Time Flight_Duration_Minutes Flight_Status Distance_Miles Price_USD Travel_Purpose Seat_Class Bags_Checked Frequent_Flyer_Status Delay_Minutes Seat_Selected 
# Data Cleaning:
#   1. Detect and handle missing values
#   2. If not too many, we can drop the rows with missing values
#   3. If too many, we can fill the missing values with mean or median
#   4. Detect and handle outliers
# Data Preprocessing: Categorical and Numerical Features --> For Categorical Features, we will use One-Hot Encoding and for Numerical Features, we will use Standardization
# Split the data into train and test sets
# Create Pipeline for Data Preprocessing (Completed in previous steps)
# Select and Train the model
# Evaluate the model and regulate it if necessary

print(data.isnull().sum()) # Frequent_Flyer_Status has 4948 missing values, so we can drop it
data.drop(columns=["Frequent_Flyer_Status"], inplace=True)

# Check for missing values after dropping columns
print("Missing values after dropping columns:")
print(data.isnull().sum())

 
selected_features = [
    "Airline", "Departure_Time","Flight_Duration_Minutes", "Flight_Status", "Distance_Miles", "Price_USD",
    "Seat_Class", "Delay_Minutes", "Seat_Selected","Travel_Purpose"
]


# Features (X) and target (y)
X = data[selected_features].copy()
y = data["Flight_Satisfaction_Score"].copy()

# Check unique values in categorical features
for col in X.select_dtypes(include=['object', 'category']):
    print(f"{col}: {data[col].unique()} --> {len(data[col].unique())} unique values")

# Encoding for "Airline" category is determined using real average values
X= pd.get_dummies(X, columns=["Airline"], prefix="Airline")

# Feature Engineering

X["Price_USD"] = -1*X["Price_USD"]   # Multiply by -1 to show neagative impact
X["Distance_Miles"] = -1*X["Distance_Miles"]
X["Flight_Duration_Minutes"] = -1*X["Flight_Duration_Minutes"]
X["Delay_Minutes"] = -1*X["Delay_Minutes"]

# Extract hour information from the Departure_Time column
X['Departure_Hour'] = pd.to_datetime(X['Departure_Time']).dt.hour

# Create categories based on time intervals
def categorize_time(hour):
    if 7 <= hour < 12:
        return "Morning"  # Between 7 AM and 12 PM
    elif 12 <= hour < 18:
        return "Afternoon"  # Between 12 PM and 6 PM
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"  # Between 12 AM and 7 AM

X['Departure_Period'] = X['Departure_Hour'].apply(categorize_time)

X = pd.get_dummies(X, columns=["Departure_Period"], prefix="Period")

# Remove unnecessary columns
X.drop(columns=["Departure_Time", "Departure_Hour"], inplace=True)

# Our new feature names are "Period_Afternoon", "Period_Morning", "Period_Night"


X = pd.get_dummies(X, columns=["Flight_Status"], prefix="Status")

X = pd.get_dummies(X, columns=["Travel_Purpose"], prefix="Purpose")

X = pd.get_dummies(X, columns=["Seat_Selected"], prefix="Seat")

# Check the first few rows to verify
print("One-Hot Encoded X:")
print(X.columns)

# Ordinal encoding for Seat_Class
seat_class_map = {
    'First': 4,           # Highest score
    'Premium Economy': 3, # Second highest score
    'Business': 2,        # Medium level
    'Economy': 1          # Lowest score
}

X['Seat_Class'] = X['Seat_Class'].map(seat_class_map)

# ^^^ Encoding is completed ^^^

# Now Start Scaling - Normalization

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler= StandardScaler()

X[numeric_features] = scaler.fit_transform(X[numeric_features])


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,         # 10% test data
    random_state=42,       # Ensures reproducibility
    shuffle=True           # Shuffle data before splitting
)

# Now let's train the data with several models and determine the best-performing one
# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100,random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,random_state=42),
    "Support Vector Regressor": SVR(),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,random_state=42)
}

# Train models and compare results
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions on the test set
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

# Print results
print("\nModel Performance Comparison:")
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}")

# Print the best-performing model
best_model = max(results, key=lambda x: results[x]["R2"])
print(f"\nBest Performing Model: {best_model}")
print(f"R2 Score: {results[best_model]['R2']:.4f}, MSE: {results[best_model]['MSE']:.4f}")
