import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# تحميل البيانات
df = pd.read_csv("Housing.csv")

# تحويل النصوص إلى أرقام
df = pd.get_dummies(df, drop_first=True)

# تقسيم البيانات
X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# عنوان التطبيق
st.title("🏠 توقع سعر المنزل")

st.write("أدخل بيانات المنزل:")

# إنشاء حقول الإدخال
input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# زر التنبؤ
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    
    st.success(f"💰 السعر المتوقع: {prediction[0]:,.0f}")