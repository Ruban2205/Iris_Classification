import pickle 
import streamlit as st
import numpy as np

scaler_file = pickle.load(open('scaler.pkl', 'rb')) 
model_file = pickle.load(open('model.pkl', 'rb'))

def pred_output(user_input):
    scaled_input = scaler_file.transform(np.array(user_input).reshape(-1,4))
    ypred = model_file.predict(scaled_input)
    return ypred[0]

def main(): 
    st.title('Iris Classification')
    st.text('Developed by - www.rubangino.in')

    st.divider()

    # Input Variables
    sepalLength = st.text_input('Enter the Sepal Length (Cm)')
    sepalWidth = st.text_input('Enter the Sepal Width in (Cm)')
    petalLength = st.text_input('Enter the Petal Length in (Cm)')
    petalWidth = st.text_input('Enter the Petal Width in (Cm)')

    if sepalLength.isalpha() or sepalWidth.isalpha() or petalLength.isalpha() or petalWidth.isalpha():
        st.error("Input must be a Numeric!", icon='🚨')
        st.error("It seems some given input was not a Numeric Value!!", icon='🤔')

    # Button to predict
    if st.button('Predict'):
        user_input = [sepalLength, sepalWidth, petalLength, petalWidth]
        make_prediction = pred_output(user_input)
        st.success(make_prediction)
    
if __name__ == '__main__':
    main()

