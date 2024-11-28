import streamlit as st 
import pandas as pd
import datawork as dw

##################### PAGE SETUP #####################
st.set_page_config(
    page_title="IRON REGRESSION QUEST",
    layout="centered", # or wide
    page_icon="📈", # choose your favorite icon
    initial_sidebar_state="expanded" # or expanded
)

##################### PAGE CONTENT #####################

LR_data = pd.DataFrame(columns=['test', 'R2', 'RMSE', 'MSE', 'MAE'])

df = dw.read_data()

st.title("📈 Iron Regression Quest 📈")
st.subheader("Linear Regression Model")
st.write("This is the study for a simple linear regression model that predicts the price of a house based on its features.")

st.subheader("TEST 1 - 100% of our data, no major teatment")

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(df.drop('price', axis=1), df['price'])
LR_data.loc[len(LR_data)] = ['1', r2, RMSE, MSE, MAE]

st.text(f"R2 = {round(r2, 4)}")
st.text(f"RMSE = {round(RMSE, 4)}")
st.text(f"MSE = {round(MSE, 4)}")
st.text(f"MAE = {round(MAE, 4)}")

####
st.subheader("TEST 2 - Data treated by removing unwanted columns and outliers.")
st.text("Columns with less than 30 unique values were removed.\n Unwanted columns with multiculinearity or unwanted distribution were also removed.")

LR_df = dw.LR_treatment(df)

for col in LR_df.columns:
    LR_df = dw.remove_outliers(LR_df, col)

X = LR_df.drop('price', axis=1)
y = LR_df["price"]

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(X, y)
LR_data.loc[len(LR_data)] = ['2', r2, RMSE, MSE, MAE]

st.text(f"Remaining columns: {LR_df.columns.to_list()}")
st.text(f"R2 = {round(r2, 4)}")
st.text(f"RMSE = {round(RMSE, 4)}")
st.text(f"MSE = {round(MSE, 4)}")
st.text(f"MAE = {round(MAE, 4)}")

####

st.subheader("TEST 3 - Normalized data from test 2")

X = dw.get_normalized_X(LR_df)
y = LR_df["price"]

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(X, y)
LR_data.loc[len(LR_data)] = ['3', r2, RMSE, MSE, MAE]

st.text(f"R2 = {round(r2, 4)}")
st.text(f"RMSE = {round(RMSE, 4)}")
st.text(f"MSE = {round(MSE, 4)}")
st.text(f"MAE = {round(MAE, 4)}")

####

st.subheader("TEST 4 - Standardized data from test 2")

X = dw.get_standardized_X(LR_df)
y = LR_df["price"]

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(X, y)
LR_data.loc[len(LR_data)] = ['4', r2, RMSE, MSE, MAE]

st.text(f"R2 = {round(r2, 4)}")
st.text(f"RMSE = {round(RMSE, 4)}")
st.text(f"MSE = {round(MSE, 4)}")
st.text(f"MAE = {round(MAE, 4)}")

####

st.subheader("TEST 5 - Removed sqft_lot and yr_built columns to check if they were hurting the model performance.")

X = LR_df.drop(['price','yr_built','sqft_lot'],axis=1)
y = LR_df["price"]

st.text(f"Remaining columns: {X.columns.to_list()}")

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(X, y)
LR_data.loc[len(LR_data)] = ['5', r2, RMSE, MSE, MAE]

st.text(f"R2 = {round(r2, 4)}")
st.text(f"RMSE = {round(RMSE, 4)}")
st.text(f"MSE = {round(MSE, 4)}")
st.text(f"MAE = {round(MAE, 4)}")

####

st.subheader("TEST 6 - Normalized data from test 4")

X = dw.get_normalized_X(LR_df.drop(['yr_built','sqft_lot'],axis=1))
y = LR_df["price"]

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(X, y)
LR_data.loc[len(LR_data)] = ['6', r2, RMSE, MSE, MAE]

st.text(f"R2 = {round(r2, 4)}")
st.text(f"RMSE = {round(RMSE, 4)}")
st.text(f"MSE = {round(MSE, 4)}")
st.text(f"MAE = {round(MAE, 4)}")

####

st.subheader("Result Analysis")

st.line_chart(LR_data, x='test', y=['R2'])
st.line_chart(LR_data, x='test', y=['RMSE'])
st.line_chart(LR_data, x='test', y=['MSE'])
st.line_chart(LR_data, x='test', y=['MAE'])


st.subheader("Conclusion")

st.text("From what we can gather for the Linear Regression model, the best results are for test 2,3 and 4, where the RMSE, MSE and MAE values are the lowest.")
st.text("Following tests present worst performance, which might be caused by the removal of too many meaningful features.")