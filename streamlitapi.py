import pickle 
import streamlit as st

scaler_file = pickle.load(open('scaler.pkl', 'rb')) 
model_file = pickle.load(open('model.pkl', 'rb'))

def main(): 
    st.title("Iris Classification - rubangino.in")

    # Input Variables
    sepalLength = st.text_input('Enter the Sepal Length (Cm)')
    sepalWidth = st.text_input('Enter the Sepal Width in (Cm)')
    petalLength = st.text_input('Enter the Petal Length in (Cm)')
    petalWidth = st.text_input('Enter the Petal Width in (Cm)')

    # Button to predict
    if st.button('Predict'):
        make_prediction = model_file.predict([[sepalLength, sepalWidth, petalLength, petalWidth]])
        st.success(make_prediction)
    
if __name__ == '__main__':
    main()

