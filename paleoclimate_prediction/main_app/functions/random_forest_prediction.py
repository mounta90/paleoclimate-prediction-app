# Import necessary libraries
import pandas as pd
import numpy as np
from django.contrib.staticfiles.storage import staticfiles_storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def random_forest_prediction():
    # Load the dataset and preposing
    csv_path = staticfiles_storage.path("main_app/Met3brAZ.csv")
    data = pd.read_csv(csv_path)
    # print(data)
    # data.fillna(data.mean(), inplace=True)
    data = pd.get_dummies(data, columns=["Sample"], drop_first=True)

    # Split the dataset into features (X) and target (y)
    X = data.drop("GSP", axis=1)
    y = data["GSP"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=50, random_state=10)

    # Train the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    #mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("PRINTING R2.....")
    print(r2)
    # print(f"R-squared (R2) Score: {r2:.4f}")
    return r2
