# basic-linear-regression
 
This project demonstrates the fundamentals of linear regression and basic regression modeling using synthetic flight passenger data. The code covers data preprocessing, feature engineering, categorical encoding, scaling, model training, and performance evaluation.

## Dataset

- **File:** `synthetic_flight_passenger_data.csv`
- **Target Variable:** `Flight_Satisfaction_Score`
- **Features:** Airline, Departure Time, Flight Duration, Flight Status, Distance, Price, Seat Class, Delay, Seat Selected, Travel Purpose, etc.

You can acces dataset from that link: 
"https://www.kaggle.com/datasets/keatonballard/synthetic-airline-passenger-and-flight-data"

## Steps

1. **Data Loading & Cleaning:**  
   - Handle missing values (drop or fill as appropriate).
   - Remove unnecessary columns.

2. **Feature Engineering:**  
   - Extract and categorize departure time.
   - Encode categorical variables (one-hot and ordinal encoding).
   - Scale numerical features.

3. **Model Training & Evaluation:**  
   - Train multiple regression models:
     - Linear Regression
     - Random Forest
     - Gradient Boosting
     - Support Vector Regressor
     - XGBoost
   - Evaluate models using MSE and R² score.

4. **Comparison:**  
   - Print and compare model performances.

## How to Run

1. Install required libraries:
    ```
    pip install pandas numpy scikit-learn seaborn matplotlib xgboost
    ```
2. Place `synthetic_flight_passenger_data.csv` in the appropriate directory.
3. Run the script:
    ```
    python BasicLineerRegression.py
    ```

## Notes

- The goal is to illustrate the basic logic of linear regression, not to achieve high accuracy.
- Model results and comparisons are printed to the console.

---

*This project is for educational and demonstration purposes only.*
