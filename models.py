from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

# Function that takes in raw data and returns:
    # X_train
    #y_train
    #X_test
    #y_test
    #X_train_processed (OHE and scaling)
    #X_test_processed (OHE and scaling)
    #train_dates (dataframe of dates in x_train)
    #test_dates (dataframe of dates in x_train)

def preprocess():
    #read in data
    data=pd.read_csv("bike-sharing_hourly.csv")
    data.head()

    # convert date column to datetime datatype
    data["dteday"]=pd.to_datetime(data["dteday"])

    ########## new features

    ## new features

    # day
    data["day"] = data["dteday"].dt.day


    # is weekend boolean
    data["is_weekend"] = np.where(data["weekday"] < 5, 0, 1)

    # period of day (morning, afternoon, evening, night)
    data["day_period"] = np.where(data['hr'] <= 5, '1', 
                                np.where(data['hr'].between(6, 11, inclusive='both'), '2',
                                         np.where(data['hr'].between(12, 17, inclusive='both'), '3',
                                                  np.where(data['hr'].between(18, 23, inclusive='both'), '4', '0'))))

    # weather factor
    data["weather_factor"] = data["windspeed"] * data["weathersit"]

    ## Temperature

    # minimum daily temperature
    data["min_daily_temp"] = data.groupby('dteday')['temp'].transform('min')

    # max daily temperature
    data["max_daily_temp"] = data.groupby('dteday')['temp'].transform('max')

    # mean daily temperature
    data["mean_daily_temp"] = data.groupby('dteday')['temp'].transform('mean')

    ## Humidity

    # minimum daily temperature
    data["min_daily_hum"] = data.groupby('dteday')['hum'].transform('min')

    # max daily temperature
    data["max_daily_hum"] = data.groupby('dteday')['hum'].transform('max')

    # mean daily temperature
    data["mean_daily_hum"] = data.groupby('dteday')['hum'].transform('mean')
    
    # rush hour
    data['is_rush_hour'] = ((data['workingday'] == 1) & (data['hr'].isin([8, 17, 18]))).astype(int)

    # night
    data['is_night'] = (data['hr'].isin([0, 1, 2, 3, 4, 5])).astype(int)

    # convert time features into their cosine, as they are cyclical
    data['cos_hour'] = data['hr'].map(
        lambda h: np.cos(2*np.pi*h/24)
    )
    data['cos_month'] = data['mnth'].map(
        lambda h: np.cos(2*np.pi*h/12)
    )
    data['cos_weekday'] = data['weekday'].map(
        lambda h: np.cos(2*np.pi*h/7)
    )
    data['cos_day'] = data['day'].map(
    lambda h: np.cos(2*np.pi*h/31)
)

    ##### dataset prep
    num_feat = ["temp", 
                "atemp", 
                "hum", 
                "cos_month", 
                "cos_hour",
                "cos_weekday",
               "weather_factor",
                "min_daily_temp",
                "max_daily_temp",
                "mean_daily_temp",
                "min_daily_hum",
                "mean_daily_hum"
               ]

    cat_feat = [ "season", 
                "yr", 
                "is_weekend",
                "day_period",
                "weathersit",
                "is_rush_hour"           
               ]

    target_feat = ["cnt"]

    data = data.sort_values(["dteday", "hr"])
    x_df = data[[*num_feat, *cat_feat]]
    x_df = x_df.reindex(sorted(x_df.columns), axis=1)
    y_df = data[[*target_feat]]
    data["datetime"] = data["dteday"] + pd.to_timedelta(data["hr"], unit="h")
    dates = data[["datetime"]]

    ##### train test split
    training_proportion = 0.9
    
    maximum_training_row = int(len(data) * training_proportion)

    # create the first dataset with the first 80% of the rows
    X_train = x_df.iloc[:maximum_training_row,:]
    y_train = y_df.iloc[:maximum_training_row,:].squeeze()

    # create the second dataset with the last 20% of the rows
    X_test = x_df.iloc[maximum_training_row:,:]
    y_test = y_df.iloc[maximum_training_row:,:].squeeze()
    
    # create x and y dates
    train_dates = dates.iloc[:maximum_training_row,:]
    test_dates = dates.iloc[maximum_training_row:,:]
    
    # create x and y dates
    train_dates = dates.iloc[:maximum_training_row,:]
    test_dates = dates.iloc[maximum_training_row:,:]
        
    ##### scaling and encoding
    num_transformer = Pipeline(steps=[('num_scaler', StandardScaler())])
    #date_transformer = Pipeline(steps=[('date_noop', FunctionTransformer(lambda x: x))])
    cat_transformer = Pipeline(steps=[('encoder', OneHotEncoder())])

    # Define the preprocessor, specifying which transformers to apply to each data type
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_feat),
        ('cat', cat_transformer, cat_feat)])

    # Fit and transform the data using the preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get the names of the columns after transformation
    num_col_names = num_feat
    cat_col_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_feat)

    # Combine the transformed data and column names into a pandas dataframe
    X_train_processed = pd.DataFrame(X_train_processed, columns=num_col_names + list(cat_col_names))
    X_test_processed = pd.DataFrame(X_test_processed, columns=num_col_names + list(cat_col_names))
    
    return X_train, y_train, X_test, y_test, X_train_processed, X_test_processed, test_dates


# Function that takes in X_train_processed, y_train, X_test_processed and  y_test and returns a dataframe of performance metrics on train and test and the predictions on the test set from a linear regression model
def linreg_model(X_train_processed, y_train, X_test_processed, y_test):
    ##### Linear Regression Model
    linreg_model = LinearRegression(
        fit_intercept=False)
    
    linreg_model.fit(X_train_processed, y_train)

    y_train_predictions = linreg_model.predict(X_train_processed)

    y_test_predictions = linreg_model.predict(X_test_processed)
    
    ###### Performance metrics
    #Create list of dictionaries with scoring metrics
    metrics_linreg = [
         {'set': 'Train', 'metric': 'R2 score', 'value': r2_score(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MSE', 'value': mean_squared_error(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MAE', 'value': mean_absolute_error(y_train, y_train_predictions)},    
         {'set': 'Test', 'metric': 'R2 score', 'value': r2_score(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MSE', 'value': mean_squared_error(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MAE', 'value': mean_absolute_error(y_test, y_test_predictions)}]

    # create metrics DataFrame
    metrics_linreg = pd.DataFrame(metrics_linreg)
        
    # Pivot the DataFrame
    metrics_linreg = metrics_linreg.pivot(index='metric', columns='set', values='value')
    metrics_linreg = metrics_linreg.reindex(columns=['Train', 'Test'])


    return metrics_linreg, y_test_predictions

# Function that takes in X_train_processed, y_train, X_test_processed and  y_test and returns a dataframe of performance metrics on train and test and the predictions on the test set from an elastic net model
def enet_model(X_train_processed, y_train, X_test_processed, y_test):
    enet_model = ElasticNet(
    alpha=0.01,
    l1_ratio=0.25,
    max_iter=1000)

    enet_model.fit(X_train_processed, y_train)

    y_train_predictions = enet_model.predict(X_train_processed)

    y_test_predictions = enet_model.predict(X_test_processed)

    # Create list of dictionaries with scoring metrics
    metrics_enet = [
         {'set': 'Train', 'metric': 'R2 score', 'value': r2_score(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MSE', 'value': mean_squared_error(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MAE', 'value': mean_absolute_error(y_train, y_train_predictions)},    
         {'set': 'Test', 'metric': 'R2 score', 'value': r2_score(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MSE', 'value': mean_squared_error(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MAE', 'value': mean_absolute_error(y_test, y_test_predictions)}]

    # Create the DataFrame
    metrics_enet = pd.DataFrame(metrics_enet)

    # Pivot the DataFrame
    metrics_enet = metrics_enet.pivot(index='metric', columns='set', values='value')
    metrics_enet = metrics_enet.reindex(columns=['Train', 'Test'])


    return metrics_enet, y_test_predictions

# Function that takes in X_train_processed, y_train, X_test_processed and  y_test and returns a dataframe of performance metrics on train and test and the predictions on the test set from a decision tree model
def dt_model(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeRegressor(
    max_depth=7,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=5)

    dt_model.fit(X_train, y_train)

    y_train_predictions = dt_model.predict(X_train)

    y_test_predictions = dt_model.predict(X_test)

    # Create list of dictionaries with scoring metrics
    metrics_dt = [
         {'set': 'Train', 'metric': 'R2 score', 'value': r2_score(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MSE', 'value': mean_squared_error(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MAE', 'value': mean_absolute_error(y_train, y_train_predictions)},    
         {'set': 'Test', 'metric': 'R2 score', 'value': r2_score(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MSE', 'value': mean_squared_error(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MAE', 'value': mean_absolute_error(y_test, y_test_predictions)}]

    # Create the DataFrame
    metrics_dt = pd.DataFrame(metrics_dt)

    # Pivot the DataFrame
    metrics_dt = metrics_dt.pivot(index='metric', columns='set', values='value')
    metrics_dt = metrics_dt.reindex(columns=['Train', 'Test'])


    return metrics_dt, y_test_predictions

# Function that takes in X_train_processed, y_train, X_test_processed and  y_test and returns a dataframe of performance metrics on train and test and the predictions on the test set from a KNN model
def knn_model(X_train_processed, y_train, X_test_processed, y_test):
    knn_model = KNeighborsRegressor(
    n_neighbors=7,
    p=1,
    weights='distance')

    knn_model.fit(X_train_processed, y_train)

    y_train_predictions = knn_model.predict(X_train_processed)

    y_test_predictions = knn_model.predict(X_test_processed)

    # Create list of dictionaries with scoring metrics
    metrics_knn = [
         {'set': 'Train', 'metric': 'R2 score', 'value': r2_score(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MSE', 'value': mean_squared_error(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MAE', 'value': mean_absolute_error(y_train, y_train_predictions)},    
         {'set': 'Test', 'metric': 'R2 score', 'value': r2_score(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MSE', 'value': mean_squared_error(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MAE', 'value': mean_absolute_error(y_test, y_test_predictions)}]

    # Create the DataFrame
    metrics_knn = pd.DataFrame(metrics_knn)

    # Pivot the DataFrame
    metrics_knn = metrics_knn.pivot(index='metric', columns='set', values='value')
    metrics_knn = metrics_knn.reindex(columns=['Train', 'Test'])


    return metrics_knn, y_test_predictions

# Function that takes in X_train_processed, y_train, X_test_processed and  y_test and returns a dataframe of performance metrics on train and test and the predictions on the test set from a random forest model
def rf_model(X_train, y_train, X_test, y_test):
    rf_model = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=7,
        min_samples_split=5,
        n_estimators=100,
        ccp_alpha=10,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    y_train_predictions = rf_model.predict(X_train)

    y_test_predictions = rf_model.predict(X_test)

    # Create list of dictionaries with scoring metrics
    metrics_rf = [
         {'set': 'Train', 'metric': 'R2 score', 'value': r2_score(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MSE', 'value': mean_squared_error(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MAE', 'value': mean_absolute_error(y_train, y_train_predictions)},    
         {'set': 'Test', 'metric': 'R2 score', 'value': r2_score(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MSE', 'value': mean_squared_error(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MAE', 'value': mean_absolute_error(y_test, y_test_predictions)}]

    # Create the DataFrame
    metrics_rf = pd.DataFrame(metrics_rf)

    # Pivot the DataFrame
    metrics_rf = metrics_rf.pivot(index='metric', columns='set', values='value')
    metrics_rf = metrics_rf.reindex(columns=['Train', 'Test'])


    return metrics_rf, y_test_predictions

# Function that takes in X_train_processed, y_train, X_test_processed and  y_test and returns a dataframe of performance metrics on train and test and the predictions on the test set from a random forest model
def gb_model(X_train, y_train, X_test, y_test):
    gb_model = GradientBoostingRegressor(
    max_depth = 5, 
    max_features = None, 
    min_samples_leaf = 1, 
    min_samples_split = 7,
    alpha = 0.1,
    n_estimators = 150)

    gb_model.fit(X_train, y_train)

    y_train_predictions = gb_model.predict(X_train)

    y_test_predictions = gb_model.predict(X_test)
    

    # Create list of dictionaries with scoring metrics
    metrics_gb = [
         {'set': 'Train', 'metric': 'R2 score', 'value': r2_score(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MSE', 'value': mean_squared_error(y_train, y_train_predictions)},    
         {'set': 'Train', 'metric': 'MAE', 'value': mean_absolute_error(y_train, y_train_predictions)},    
         {'set': 'Test', 'metric': 'R2 score', 'value': r2_score(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MSE', 'value': mean_squared_error(y_test, y_test_predictions)},    
         {'set': 'Test', 'metric': 'MAE', 'value': mean_absolute_error(y_test, y_test_predictions)}]

    # Create the DataFrame
    metrics_gb = pd.DataFrame(metrics_gb)

    # Pivot the DataFrame
    metrics_gb = metrics_gb.pivot(index='metric', columns='set', values='value')
    metrics_gb = metrics_gb.reindex(columns=['Train', 'Test'])
    
    feature_importance = gb_model.feature_importances_
    sorted_idx = feature_importance.argsort()
    estimators = gb_model.estimators_
    feature_names=gb_model.feature_names_in_
    
    importances_df = pd.DataFrame({'Feature': gb_model.feature_names_in_, 'Importance': gb_model.feature_importances_})
    importances_df = importances_df.sort_values("Importance", ascending=False)
    

    return metrics_gb, y_test_predictions, feature_importance, sorted_idx, estimators, feature_names


# Function takes array of predictions, dataframe of dates, and series of target values and outputs a dataframe with dates, actual target and predictions
def create_predictions_df(model_predictions, test_dates, y_test):
    
    # predictions and y_test to df
    y_test_predictions = pd.DataFrame(model_predictions)
    y_test = pd.DataFrame(y_test)

    # create df with dates and actual count
    predictions_df = pd.merge(test_dates, y_test, left_index=True, right_index=True, how='outer')

    # rest index
    predictions_df = predictions_df.reset_index().drop(columns="index")

    # join df with predicted count from model
    predictions_df = pd.merge(predictions_df, y_test_predictions, left_index=True, right_index=True, how='outer')

    # rename predictions column
    predictions_df.rename(columns={"cnt": "actual_count",
                                   0: "predicted_count"}, inplace=True)

    # change prediction from float to int
    predictions_df["predicted_count"] = predictions_df["predicted_count"].astype(int)
    
    return predictions_df
