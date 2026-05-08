import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🏠 House Price Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

@st.cache_resource
def train_model(df):
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("price", axis=1)
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns

# تحميل البيانات مرة واحدة فقط
df = load_data()

# تدريب مرة واحدة فقط
model, columns = train_model(df)

st.write("Enter house data:")

input_data = {}
for col in columns:
    input_data[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)
    st.success(f"Price: {prediction[0]:,.0f}")
