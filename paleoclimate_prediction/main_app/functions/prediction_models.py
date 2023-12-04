# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import r2_score

# Import Django library for static files usage.
from django.contrib.staticfiles.storage import staticfiles_storage

# Import Joblib, a package allowing you to save and load saved models.
import joblib


def model_comparison_before_hpo() -> dict:
    # Define which parameter to predict
    # ENTHAL : 1
    # MAT : 2
    # SH : 3
    # GSP : 4
    # RH : 5

    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values
    y = data.iloc[:, -2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Algorithm Selection
    models = [
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('LR', LinearRegression()),
        ('DT', DecisionTreeRegressor())
    ]

    # Cross-Validation and Model Evaluation
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = scores

    # Final Selection
    final_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        final_scores[name] = r2

    # print("Performance Before HPO")
    #
    # best_model = max(final_scores, key=final_scores.get)
    # for name, r2 in final_scores.items():
    #     print(f"{name} R-squared: {r2}")
    #
    # print("Best Model:", best_model, final_scores[best_model])
    #
    # print("----------------------------")

    results_dict = {}
    for model, r2 in final_scores.items():
        results_dict[model] = r2

    return results_dict


def model_comparison_after_hpo() -> dict:
    # Define which parameter to predict
    # ENTHAL : 1
    # MAT : 2
    # SH : 3
    # GSP : 4
    # RH : 5

    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values
    y = data.iloc[:, -2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Algorithm Selection
    models = [
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('LR', LinearRegression()),
        ('DT', DecisionTreeRegressor())
    ]

    # Cross-Validation and Model Evaluation
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = scores

    # Final Selection
    final_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        final_scores[name] = r2

    # print("----------------------------")

    # -----------------------------------
    # Optimized SVR Prediction
    # -----------------------------------
    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_mat")
    best_svr_model_mat = joblib.load(model_path)

    svr_pred = best_svr_model_mat.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)

    # -----------------------------------
    # Optimized Random Forest Prediction
    # -----------------------------------
    model_path = staticfiles_storage.path("main_app/saved_models/best_rf_model_mat")
    best_rf_model_mat = joblib.load(model_path)

    rf_pred = best_rf_model_mat.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    # -----------------------------------
    # Optimized Linear Regression Prediction
    # -----------------------------------
    model_path = staticfiles_storage.path("main_app/saved_models/best_lr_model_mat")
    best_lr_model_mat = joblib.load(model_path)

    lr_pred = best_lr_model_mat.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    # -----------------------------------
    # Optimized Decision Tree Prediction
    # -----------------------------------
    model_path = staticfiles_storage.path("main_app/saved_models/best_dt_model_mat")
    best_dt_model_mat = joblib.load(model_path)

    dt_pred = best_dt_model_mat.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)

    # -----------------------------------
    # Final Selection after HPO
    # -----------------------------------
    final_scores = {
        'RF': rf_r2,
        'SVR': svr_r2,
        'LR': lr_r2,
        'DT': dt_r2
    }
    # print("Performance After HPO")
    # best_model = max(final_scores, key=final_scores.get)
    #
    # for name, r2 in final_scores.items():
    #     print(f"{name} R-squared: {r2}")
    #
    # print("Best Model:", best_model, final_scores[best_model])
    #
    # print("----------------------------")

    results_dict = {}
    for model, r2 in final_scores.items():
        results_dict[model] = r2

    return results_dict


def train_and_save_mat_model():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values
    y = data.iloc[:, -2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Algorithm Selection
    models = [
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('LR', LinearRegression()),
        ('DT', DecisionTreeRegressor())
    ]

    # Cross-Validation and Model Evaluation
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = scores

    # Final Selection
    final_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        final_scores[name] = r2

    print("----------------------------")
    # Hyperparameter Optimization
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    svr_param_grid = {
        'C': [0.1, 0.8, 2, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    svr_optimized = GridSearchCV(SVR(), param_grid=svr_param_grid, cv=5, scoring='r2')
    svr_optimized.fit(X_train, y_train)
    best_svr_model = svr_optimized.best_estimator_

    joblib.dump(best_svr_model, "../static/main_app/saved_models/best_svr_model_mat")

    svr_pred = best_svr_model.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)

    # Hyperparameter Optimization for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 150, 100, 200],
        'max_depth': [None, 10, 18, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    rf_optimized = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, cv=5, scoring='r2')
    rf_optimized.fit(X_train, y_train)
    best_rf_model = rf_optimized.best_estimator_

    joblib.dump(best_rf_model, "../static/main_app/saved_models/best_rf_model_mat")

    rf_pred = best_rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    # Hyperparameter Optimization for Linear Regression
    lr_param_grid = {
        'fit_intercept': [True, False]
    }

    lr_optimized = GridSearchCV(LinearRegression(), param_grid=lr_param_grid, cv=5, scoring='r2')
    lr_optimized.fit(X_train, y_train)
    best_lr_model = lr_optimized.best_estimator_

    joblib.dump(best_lr_model, "../static/main_app/saved_models/best_lr_model_mat")

    lr_pred = best_lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    # Hyperparameter Optimization for Decision Tree
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30, 50, 100],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 7]
    }

    dt_optimized = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=5, scoring='r2')
    dt_optimized.fit(X_train, y_train)
    best_dt_model = dt_optimized.best_estimator_

    joblib.dump(best_dt_model, "../static/main_app/saved_models/best_dt_model_mat")

    dt_pred = best_dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)

    # Final Selection after HPO
    final_scores = {
        'Random Forest': rf_r2,
        'SVR': svr_r2,
        'Linear Regression': lr_r2,
        'Decision Tree': dt_r2
    }
    print("Perfomance After HPO")
    best_model = max(final_scores, key=final_scores.get)

    for name, r2 in final_scores.items():
        print(f"{name} R-squared: {r2}")

    # print("Selected Algorithms:", models)
    print("Best Model:", best_model, final_scores[best_model])


def train_and_save_sh_model():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values
    y = data.iloc[:, -3].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Algorithm Selection
    models = [
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('LR', LinearRegression()),
        ('DT', DecisionTreeRegressor())
    ]

    # Cross-Validation and Model Evaluation
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = scores

    # Final Selection
    final_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        final_scores[name] = r2

    print("----------------------------")
    # Hyperparameter Optimization
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    svr_param_grid = {
        'C': [0.1, 0.8, 2, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    svr_optimized = GridSearchCV(SVR(), param_grid=svr_param_grid, cv=5, scoring='r2')
    svr_optimized.fit(X_train, y_train)
    best_svr_model = svr_optimized.best_estimator_

    joblib.dump(best_svr_model, "../static/main_app/saved_models/best_svr_model_sh")

    svr_pred = best_svr_model.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)

    # Hyperparameter Optimization for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 150, 100, 200],
        'max_depth': [None, 10, 18, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    rf_optimized = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, cv=5, scoring='r2')
    rf_optimized.fit(X_train, y_train)
    best_rf_model = rf_optimized.best_estimator_

    joblib.dump(best_rf_model, "../static/main_app/saved_models/best_rf_model_sh")

    rf_pred = best_rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    # Hyperparameter Optimization for Linear Regression
    lr_param_grid = {
        'fit_intercept': [True, False]
    }

    lr_optimized = GridSearchCV(LinearRegression(), param_grid=lr_param_grid, cv=5, scoring='r2')
    lr_optimized.fit(X_train, y_train)
    best_lr_model = lr_optimized.best_estimator_

    joblib.dump(best_lr_model, "../static/main_app/saved_models/best_lr_model_sh")

    lr_pred = best_lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    # Hyperparameter Optimization for Decision Tree
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30, 50, 100],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 7]
    }

    dt_optimized = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=5, scoring='r2')
    dt_optimized.fit(X_train, y_train)
    best_dt_model = dt_optimized.best_estimator_

    joblib.dump(best_dt_model, "../static/main_app/saved_models/best_dt_model_sh")

    dt_pred = best_dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)

    # Final Selection after HPO
    final_scores = {
        'Random Forest': rf_r2,
        'SVR': svr_r2,
        'Linear Regression': lr_r2,
        'Decision Tree': dt_r2
    }
    print("Perfomance After HPO")
    best_model = max(final_scores, key=final_scores.get)

    for name, r2 in final_scores.items():
        print(f"{name} R-squared: {r2}")

    # print("Selected Algorithms:", models)
    print("Best Model:", best_model, final_scores[best_model])


def train_and_save_gsp_model():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values
    y = data.iloc[:, -4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Algorithm Selection
    models = [
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('LR', LinearRegression()),
        ('DT', DecisionTreeRegressor())
    ]

    # Cross-Validation and Model Evaluation
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = scores

    # Final Selection
    final_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        final_scores[name] = r2

    print("----------------------------")
    # Hyperparameter Optimization
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    svr_param_grid = {
        'C': [0.1, 0.8, 2, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    svr_optimized = GridSearchCV(SVR(), param_grid=svr_param_grid, cv=5, scoring='r2')
    svr_optimized.fit(X_train, y_train)
    best_svr_model = svr_optimized.best_estimator_

    joblib.dump(best_svr_model, "../static/main_app/saved_models/best_svr_model_gsp")

    svr_pred = best_svr_model.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)

    # Hyperparameter Optimization for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 150, 100, 200],
        'max_depth': [None, 10, 18, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    rf_optimized = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, cv=5, scoring='r2')
    rf_optimized.fit(X_train, y_train)
    best_rf_model = rf_optimized.best_estimator_

    joblib.dump(best_rf_model, "../static/main_app/saved_models/best_rf_model_gsp")

    rf_pred = best_rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    # Hyperparameter Optimization for Linear Regression
    lr_param_grid = {
        'fit_intercept': [True, False]
    }

    lr_optimized = GridSearchCV(LinearRegression(), param_grid=lr_param_grid, cv=5, scoring='r2')
    lr_optimized.fit(X_train, y_train)
    best_lr_model = lr_optimized.best_estimator_

    joblib.dump(best_lr_model, "../static/main_app/saved_models/best_lr_model_gsp")

    lr_pred = best_lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    # Hyperparameter Optimization for Decision Tree
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30, 50, 100],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 7]
    }

    dt_optimized = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=5, scoring='r2')
    dt_optimized.fit(X_train, y_train)
    best_dt_model = dt_optimized.best_estimator_

    joblib.dump(best_dt_model, "../static/main_app/saved_models/best_dt_model_gsp")

    dt_pred = best_dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)

    # Final Selection after HPO
    final_scores = {
        'Random Forest': rf_r2,
        'SVR': svr_r2,
        'Linear Regression': lr_r2,
        'Decision Tree': dt_r2
    }
    # print("Perfomance After HPO")
    # best_model = max(final_scores, key=final_scores.get)

    # for name, r2 in final_scores.items():
    #     print(f"{name} R-squared: {r2}")
    #
    # # print("Selected Algorithms:", models)
    # print("Best Model:", best_model, final_scores[best_model])


def train_and_save_rh_model():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values
    y = data.iloc[:, -5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Algorithm Selection
    models = [
        ('RF', RandomForestRegressor()),
        ('SVR', SVR()),
        ('LR', LinearRegression()),
        ('DT', DecisionTreeRegressor())
    ]

    # Cross-Validation and Model Evaluation
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = scores

    # Final Selection
    final_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        final_scores[name] = r2

    print("----------------------------")
    # Hyperparameter Optimization
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    svr_param_grid = {
        'C': [0.1, 0.8, 2, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    svr_optimized = GridSearchCV(SVR(), param_grid=svr_param_grid, cv=5, scoring='r2')
    svr_optimized.fit(X_train, y_train)
    best_svr_model = svr_optimized.best_estimator_

    joblib.dump(best_svr_model, "../static/main_app/saved_models/best_svr_model_rh")

    svr_pred = best_svr_model.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)

    # Hyperparameter Optimization for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 150, 100, 200],
        'max_depth': [None, 10, 18, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    rf_optimized = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, cv=5, scoring='r2')
    rf_optimized.fit(X_train, y_train)
    best_rf_model = rf_optimized.best_estimator_

    joblib.dump(best_rf_model, "../static/main_app/saved_models/best_rf_model_rh")

    rf_pred = best_rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    # Hyperparameter Optimization for Linear Regression
    lr_param_grid = {
        'fit_intercept': [True, False]
    }

    lr_optimized = GridSearchCV(LinearRegression(), param_grid=lr_param_grid, cv=5, scoring='r2')
    lr_optimized.fit(X_train, y_train)
    best_lr_model = lr_optimized.best_estimator_

    joblib.dump(best_lr_model, "../static/main_app/saved_models/best_lr_model_rh")

    lr_pred = best_lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    # Hyperparameter Optimization for Decision Tree
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30, 50, 100],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 7]
    }

    dt_optimized = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=5, scoring='r2')
    dt_optimized.fit(X_train, y_train)
    best_dt_model = dt_optimized.best_estimator_

    joblib.dump(best_dt_model, "../static/main_app/saved_models/best_dt_model_rh")

    dt_pred = best_dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)

    # Final Selection after HPO
    final_scores = {
        'Random Forest': rf_r2,
        'SVR': svr_r2,
        'Linear Regression': lr_r2,
        'Decision Tree': dt_r2
    }
    print("Perfomance After HPO")
    best_model = max(final_scores, key=final_scores.get)

    for name, r2 in final_scores.items():
        print(f"{name} R-squared: {r2}")

    # print("Selected Algorithms:", models)
    print("Best Model:", best_model, final_scores[best_model])


def make_enthalpy_prediction_penn():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model")
    best_svr_model_enthalpy = joblib.load(model_path)

    # Predict the sample from Arendtsville, Pennsylvania
    sample = X[6].reshape(1, -1)
    enthalpy_prediction = best_svr_model_enthalpy.predict(sample)[0]

    return round(enthalpy_prediction, 2)


def make_enthalpy_prediction_wyo():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/kissinger-lakes-wyoming-sample.csv")
    features = pd.read_csv(csv_path)

    sample = features.values[0][1:].reshape(1, -1)

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model")
    best_svr_model_enthalpy = joblib.load(model_path)

    # Predict the sample from Kissinger Lakes, Wyoming
    enthalpy_prediction = best_svr_model_enthalpy.predict(sample)[0]

    return round(enthalpy_prediction, 2)


def make_mat_prediction_penn():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_mat")
    best_svr_model_mat = joblib.load(model_path)

    # Predict the sample from Arendtsville, Pennsylvania
    sample = X[6].reshape(1, -1)
    mat_prediction = best_svr_model_mat.predict(sample)[0]

    return round(mat_prediction, 1)


def make_mat_prediction_wyo():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/kissinger-lakes-wyoming-sample.csv")
    features = pd.read_csv(csv_path)

    sample = features.values[0][1:].reshape(1, -1)

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_mat")
    best_svr_model_mat = joblib.load(model_path)

    # Predict the sample from Kissinger Lakes, Wyoming
    mat_prediction = best_svr_model_mat.predict(sample)[0]

    return round(mat_prediction, 1)


def make_sh_prediction_penn():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_sh")
    best_svr_model_sh = joblib.load(model_path)

    # Predict the sample from Arendtsville, Pennsylvania
    sample = X[6].reshape(1, -1)
    sh_prediction = best_svr_model_sh.predict(sample)[0]

    return round(sh_prediction, 1)


def make_sh_prediction_wyo():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/kissinger-lakes-wyoming-sample.csv")
    features = pd.read_csv(csv_path)

    sample = features.values[0][1:].reshape(1, -1)

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_sh")
    best_svr_model_sh = joblib.load(model_path)

    # Predict the sample from Kissinger Lakes, Wyoming
    sh_prediction = best_svr_model_sh.predict(sample)[0]

    return round(sh_prediction, 1)


def make_gsp_prediction_penn():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_gsp")
    best_svr_model_gsp = joblib.load(model_path)

    # Predict the sample from Arendtsville, Pennsylvania
    sample = X[6].reshape(1, -1)
    gsp_prediction = best_svr_model_gsp.predict(sample)[0]

    return round(gsp_prediction, 0)


def make_gsp_prediction_wyo():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/kissinger-lakes-wyoming-sample.csv")
    features = pd.read_csv(csv_path)

    sample = features.values[0][1:].reshape(1, -1)

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_gsp")
    best_svr_model_gsp = joblib.load(model_path)

    # Predict the sample from Kissinger Lakes, Wyoming
    gsp_prediction = best_svr_model_gsp.predict(sample)[0]

    return round(gsp_prediction, 0)


def make_rh_prediction_penn():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/data.csv")
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-5].values

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_rh")
    best_svr_model_rh = joblib.load(model_path)

    # Predict the sample from Arendtsville, Pennsylvania
    sample = X[6].reshape(1, -1)
    rh_prediction = best_svr_model_rh.predict(sample)[0]

    return round(rh_prediction, 0)


def make_rh_prediction_wyo():
    # Import Data
    csv_path = staticfiles_storage.path("main_app/data/kissinger-lakes-wyoming-sample.csv")
    features = pd.read_csv(csv_path)

    sample = features.values[0][1:].reshape(1, -1)

    model_path = staticfiles_storage.path("main_app/saved_models/best_svr_model_rh")
    best_svr_model_rh = joblib.load(model_path)

    # Predict the sample from Kissinger Lakes, Wyoming
    rh_prediction = best_svr_model_rh.predict(sample)[0]

    return round(rh_prediction, 0)
