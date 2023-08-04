import pickle 
import streamlit as st

scaler_file = pickle.load(open('scaler.pkl', 'rb')) 
model_file = pickle.load(open('model.pkl'))

def main(): 
    st.title("Iris Classification - rubangino.in")

    
if __name__ == '__main__':
    main()