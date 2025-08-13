import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival.")

# Function to get user input
def user_input_features():
    Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    Sex = st.selectbox("Sex", ['male', 'female'])
    Age = st.slider("Age", 0, 80, 25)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
    Fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
    Embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
    Title = st.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

    # Encoding
    Sex = 1 if Sex == 'male' else 0
    Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]
    Title = {'Mr': 3, 'Miss': 1, 'Mrs': 2, 'Master': 0, 'Rare': 4}[Title]

    data = {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked,
        'Title': Title
    }
    return pd.DataFrame(data, index=[0])

# Get input data
input_df = user_input_features()

st.subheader("Passenger Data")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display result
st.subheader("Prediction")
st.write("🎯 Survived" if prediction[0] == 1 else "💀 Did Not Survive")

st.subheader("Prediction Probability")
st.write(f"Survival Probability: {prediction_proba[0][1]:.2f}")





