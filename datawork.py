import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
def read_data():
    df = pd.read_csv("king_ country_ houses_aa.csv")

    df = df.drop(["date"], axis=1)

    price = df.pop("price")
    df["price"] = price

    return df

def remove_outliers(df,col):

  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)

  # Compute IQR
  IQR = Q3 - Q1

  # Define bounds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Filter the dataframe
  return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


def linear_regression_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  print(f'100% of our data: {len(X)}.')
  print(f'70% for training data: {len(X_train)}.')
  print(f'30% for test data: {len(X_test)}.')

  model = LinearRegression()
  model.fit(X_train,y_train)

  predictions = model.predict(X_test)

  r2 = r2_score(y_test, predictions)
  RMSE = root_mean_squared_error(y_test, predictions)
  MSE = mean_squared_error(y_test, predictions)
  MAE = mean_absolute_error(y_test, predictions)

  #Printing the results
  print("R2 = ", round(r2, 4))
  print("RMSE = ", round(RMSE, 4))
  print("MSE =  ", round(MSE, 4))
  print("MAE = ", round(MAE, 4))

  return [r2, RMSE, MSE, MAE, predictions, y_test]

def ridge_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  print(f'100% of our data: {len(X)}.')
  print(f'70% for training data: {len(X_train)}.')
  print(f'30% for test data: {len(X_test)}.')

  ridge = Ridge()
  ridge.fit(X_train, y_train)

  predictions = ridge.predict(X_test)

  r2 = r2_score(y_test, predictions)
  RMSE = root_mean_squared_error(y_test, predictions)
  MSE = mean_squared_error(y_test, predictions)
  MAE = mean_absolute_error(y_test, predictions)

  #Printing the results
  print("R2 = ", round(r2, 4))
  print("RMSE = ", round(RMSE, 4))
  print("MSE =  ", round(MSE, 4))
  print("MAE = ", round(MAE, 4))

  return [r2, RMSE, MSE, MAE, predictions, y_test]


def lasso_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  print(f'100% of our data: {len(X)}.')
  print(f'70% for training data: {len(X_train)}.')
  print(f'30% for test data: {len(X_test)}.')

  lasso = Lasso()
  lasso.fit(X_train, y_train)

  predictions = lasso.predict(X_test)

  r2 = r2_score(y_test, predictions)
  RMSE = root_mean_squared_error(y_test, predictions)
  MSE = mean_squared_error(y_test, predictions)
  MAE = mean_absolute_error(y_test, predictions)

  #Printing the results
  print("R2 = ", round(r2, 4))
  print("RMSE = ", round(RMSE, 4))
  print("MSE =  ", round(MSE, 4))
  print("MAE = ", round(MAE, 4))

  return [r2, RMSE, MSE, MAE, predictions, y_test]

def dtmodel_expensive_houses(X_train,X_test,y_train,y_test):
  
  tree = DecisionTreeRegressor()
  tree.fit(X_train, y_train)

  predictions = tree.predict(X_test)

  r2 = r2_score(y_test, predictions)
  RMSE = root_mean_squared_error(y_test, predictions)
  MSE = mean_squared_error(y_test, predictions)
  MAE = mean_absolute_error(y_test, predictions)
  
  return [r2, RMSE, MSE, MAE, predictions, y_test]
  
def decision_tree_model(X, y, test_size):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
  print(f'100% of our data: {len(X)}.')
  print(f'70% for training data: {len(X_train)}.')
  print(f'30% for test data: {len(X_test)}.')

  tree = DecisionTreeRegressor()
  tree.fit(X_train, y_train)

  predictions = tree.predict(X_test)

  r2 = r2_score(y_test, predictions)
  RMSE = root_mean_squared_error(y_test, predictions)
  MSE = mean_squared_error(y_test, predictions)
  MAE = mean_absolute_error(y_test, predictions)

  #Printing the results
  print("R2 = ", round(r2, 4))
  print("RMSE = ", round(RMSE, 4))
  print("MSE =  ", round(MSE, 4))
  print("MAE = ", round(MAE, 4))

  return [r2, RMSE, MSE, MAE, predictions, y_test]

def LR_treatment(df):
    df_reg = df

    for col in df.columns:
        n = len(df[col].value_counts().unique())
        if n < 30:
            df_reg = df_reg.drop(col, axis=1)

    df_reg = df_reg.drop("zipcode", axis=1)
    df_reg = df_reg.drop("long", axis=1)
    df_reg = df_reg.drop("sqft_above", axis=1)
    df_reg = df_reg.drop("sqft_living15", axis=1)
    df_reg = df_reg.drop("sqft_lot15", axis=1)

    return df_reg


def DT_treatment(df):
  dt = df.drop([
    'id', # uniform and unnecessary
    'sqft_above' # multicolinearity and redundant
  ], axis=1)
  
  dt['view'] = dt['view'].apply(lambda x: 1 if x != 0 else 0) # 0 : not visited 1: visited
  dt['yr_renovated'] = dt['yr_renovated'].apply(lambda x: 1 if x != 0 else 0) # 0: not renovated 1: renovated
  
  return dt


def ridge_treatment(df_ridge):
  df_ridge = df_ridge.drop("zipcode", axis=1)
  df_ridge = df_ridge.drop("yr_built", axis=1)
  df_ridge = df_ridge.drop("long", axis=1)
  df_ridge = df_ridge.drop("sqft_above", axis=1)
  df_ridge = df_ridge.drop("sqft_living15", axis=1)
  df_ridge = df_ridge.drop("sqft_lot15", axis=1)
  df_ridge = df_ridge.drop("lat", axis = 1)
  df_ridge = df_ridge.drop("condition", axis = 1)
  df_ridge = df_ridge.drop("sqft_lot", axis=1)
  
  for col in df_ridge.columns:
    if df_ridge[col].dtype != 'object':
        df_ridge = remove_outliers(df_ridge, col)
        
  return df_ridge


def lasso_treatment(df_lasso):
  
  for col in df_lasso.columns:
    n = len(df_lasso[col].value_counts().unique())
    print(col, n)
    if n < 30:
        df_lasso = df_lasso.drop(col, axis=1)
  
  df_lasso = df_lasso.drop("zipcode", axis=1)
  df_lasso = df_lasso.drop("long", axis=1)
  df_lasso = df_lasso.drop("sqft_above", axis=1)
  df_lasso = df_lasso.drop("sqft_living15", axis=1)
  df_lasso = df_lasso.drop("sqft_lot15", axis=1)
  
  for col in df_lasso.columns:
    df_lasso = remove_outliers(df_lasso, col)
    
  return df_lasso


def get_normalized_X(df):
  df = df.drop("price", axis=1)

  scaler = MinMaxScaler()
  X_norm = df.astype(int).copy()

  scaler.fit(X_norm)

  X_norm = pd.DataFrame(scaler.transform(X_norm), columns=X_norm.columns)
  print(X_norm.head(5))

  return X_norm

def get_standardized_X(df):
  df = df.drop("price", axis=1)

  scaler = StandardScaler()
  X_norm = df.astype(int).copy()

  scaler.fit(X_norm)

  X_norm = pd.DataFrame(scaler.transform(X_norm), columns=X_norm.columns)
  print(X_norm.head(5))

  return X_norm


