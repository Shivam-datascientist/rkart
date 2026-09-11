import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

    "Marks": [35, 40, 45, 50, 55, 60, 65, 70, 80, 90,
              92, 95, 96, 98, 99, 100, 100, 100, 100, 100]
}

# DataFrame
df = pd.DataFrame(data)

# Title
st.title("📚 Student Marks Prediction")

# Show dataset
st.subheader("Student Dataset")
st.dataframe(df)

# Features and target
X = df[["Hours"]]
y = df["Marks"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# User input
hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

# Prediction
if st.button("Predict Marks"):

    prediction = model.predict(
        pd.DataFrame({"Hours": [hours]})
    )

    st.success(f"Predicted Marks: {prediction[0]:.2f}")
