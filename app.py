

import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('iris_flower.pkl', 'rb'))

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    input=np.array([[sepal_length, sepal_width, petal_length, petal_width]]).astype(np.float64)
    prediction=model.predict(input)
    return prediction

def main():

    st.title("IRIS FLOWER")
    
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Iris Flower Classification </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    sepal_length = st.slider("sepal_length", 0.0, 10.0)
    sepal_width = st.slider("sepal_width", 0.0, 10.0)
    petal_length = st.slider("petal_length", 0.0, 10.0)
    petal_width = st.slider("petal_width", 0.0, 10.0)

    
    setosa_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:white;text-align:center;"> Flower Is Setosa </h2>
       </div>
    """
    
    versicolor_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Flower Is Versicolor </h2>
       </div>
    """
    virginica_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Flower Is Virginica </h2>
       </div>
    """


    if st.button("Predict"):
        output = predict_flower(sepal_length, sepal_width, petal_length, petal_width)
        
        if output == 0:
            st.markdown(setosa_html,unsafe_allow_html=True)
        elif output == 1:
            st.markdown(versicolor_html, unsafe_allow_html=True)
        else:
            st.markdown(virginica_html,unsafe_allow_html=True)

        

if __name__ =='__main__':
    main()

