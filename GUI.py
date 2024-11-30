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
st.text("Columns with discrete data were removed.\n Unwanted columns with multiculinearity or unwanted distribution were also removed.")

LR_df = dw.LR_treatment(df)

for col in LR_df.columns:
    LR_df = dw.remove_outliers(LR_df, col)

X = LR_df.drop('price', axis=1)
y = LR_df["price"]

r2, RMSE, MSE, MAE, predictions, y_test = dw.linear_regression_model(X, y)
LR_data.loc[len(LR_data)] = ['2', r2, RMSE, MSE, MAE]

best_LR = ["Linear Regression", r2, RMSE, MSE, MAE]
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

st.subheader("TEST 6 - Normalized data from test 5")

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


st.subheader("Outcome:")

st.text("From what we can gather for the Linear Regression model, the best results are for test 2,3 and 4, where the RMSE, MSE and MAE values are the lowest.")
st.text("Following tests present worst performance, which might be caused by the removal of too many meaningful features.")


st.subheader("Models Comparison")

data_comparison = pd.DataFrame(columns=['model', 'R2', 'RMSE', 'MSE', 'MAE'])


######### Linear Regression ###################

data_comparison.loc[len(data_comparison)] = best_LR


########### Decision Tree #####################

dt = dw.read_data()
dt = dw.DT_treatment(df)

for col in dt.columns:
    dt = dw.remove_outliers(dt, col)
    
X_dt = dt.drop('price', axis=1)
y_dt = dt['price']

[r2, RMSE, MSE, MAE, predictions, y_test] = dw.decision_tree_model(X_dt, y_dt, test_size=0.1)

data_comparison.loc[len(data_comparison)] = ['Decision Tree', r2, RMSE, MSE, MAE]


########### Ridge Regression #####################


df_ridge = dw.read_data()

df_ridge = dw.LR_treatment(df_ridge)

X_ridge = df_ridge.drop('price', axis=1) #features
y_ridge = df_ridge['price'] #target

[r2, RMSE, MSE, MAE, predictions, y_test] = dw.ridge_model(X_dt, y_dt)
data_comparison.loc[len(data_comparison)] = ['Ridge Regression', r2, RMSE, MSE, MAE]


## lasso Regression

df_lasso = dw.read_data()
df_lasso = dw.lasso_treatment(df_lasso)

lasso_pipeline, r2, RMSE, MSE, MAE = dw.lasso_pipeline(df_lasso, target_column="price")

data_comparison.loc[len(data_comparison)] = ['Lasso Regression', r2, RMSE, MSE, MAE]



############ Visualization ################

st.bar_chart(data_comparison, x='model', y=['R2'], color='model')
st.bar_chart(data_comparison, x='model', y=['RMSE'], color='model')
st.bar_chart(data_comparison, x='model', y=['MSE'], color='model')
st.bar_chart(data_comparison, x='model', y=['MAE'], color='model')


st.subheader("> 650K House Price Prediction")

st.text("Given that the Decision Tree model is the one that presented the best results, we will be using it to predict the price of expensive houses.")

houses_df = dw.read_data()

high_df = houses_df[houses_df.price >= 650000]

low_df = pd.concat( [houses_df[houses_df.price < 650000],high_df.sample(2500, random_state=42)])

high_df = dw.DT_treatment(high_df)

low_df = dw.DT_treatment(low_df)

X_train = low_df.drop('price', axis=1)
y_train = low_df['price']
X_test = high_df.drop('price', axis=1)
y_test = high_df['price']

[r2, RMSE, MSE, MAE, predictions, y_test] = dw.dtmodel_expensive_houses(X_train,X_test,y_train,y_test)

best_DT = pd.DataFrame(columns=['R2', 'RMSE', 'MSE', 'MAE'])
best_DT = [r2, RMSE, MSE, MAE]

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

card1 = col1.container(border=True)
card2 = col2.container(border=True)
card3 = col3.container(border=True)
card4 = col4.container(border=True)


card1.metric("R2", round(r2, 3),)
card2.metric("RMSE", round(RMSE, 1))
card3.metric("MSE", round(MSE, 1))
card4.metric("MAE", round(MAE, 1))

print(high_df.price.median())

