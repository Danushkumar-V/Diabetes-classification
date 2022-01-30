import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('predic_model.pkl','rb'))

@st.cache(allow_output_mutation=True)
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


header_content = st.container()
dataset_descrb = st.container()
prediction = st.container()

with header_content:
    st.title('HELLO GUYS, WELCOME TO MY END-TO-END MACHINE LEARNING PROJECT!!')
    first_para = '<p style="font-family:Courier; color:Black; font-size: 20px;">In this project I have worked on Diabetes dataset and trained a model that can predict for a patient with diabetes or not...</p>'
    st.markdown(first_para, unsafe_allow_html=True)

with dataset_descrb:
    st.header('Diabetes patients dataset!')
    second_para = '<p style="font-family:Courier; color:Black; font-size: 20px;">Lets see the insights of the dataset by visualizing the dataset...</p>'
    st.markdown(second_para, unsafe_allow_html=True)
    df = load_data("diabetes_dataset.csv")
    def clas(value):
        if value == 'tested_negative':
            return 0
        elif value == 'tested_positive':
            return 1
    df['class'] = df['class'].apply(clas)


    st.area_chart(data=df.head(20), width=1000, height=300, use_container_width=True)
    with st.expander("See explanation"):
     st.write("""
         The above chart is derived using the below given data set...
     """)
     st.dataframe(df)

with prediction:
    st.header(""" Let's predict :sunglasses: """)
    third_para = '<p style="font-family:Courier; color:Black; font-size: 20px;">Enter the below values and lets check whether you are diagnosed with diabetes are not :)</p>'
    st.markdown(third_para, unsafe_allow_html=True)
    a, b = st.columns(2)
    preg = a.slider('Select the number of time you have been pregnant:',min_value=0, max_value=20,value=0, step=1 )
    plas = b.text_input('Enter the plasma glucose concentration in an oral glucose tolerance test:', 100)
    plas = int(plas)
    pres = b.slider('Select the diastolic blood pressure (mm Hg):',min_value=0, max_value=130, value=0, step=2)
    skin = a.text_input('Enter the triceps skin fold thickness (mm):', 0)
    skin = int(skin)
    insu = a.text_input('Enter the 2-Hour serum insulin (mu U/ml):', 0)
    insu = int(insu)
    weight = a.text_input('Enter the weight of the person in kilogram (kg):',0)
    height = b.text_input('Enter the height of the person in metres (cm):',1)
    weight = float(weight)
    height = float(height)
    height = height/100
    BMI = weight/height**2
    pedi = b.text_input('Enter the diabetes pedigree function:',0)
    pedi = float(pedi)
    age = a.text_input('Enter the age of the person (years):',0)
    age = int(age)
    new_data = [[preg ,plas ,pres ,skin ,insu ,BMI  ,pedi ,age]]
    predict_value = model.predict(new_data)
    result = st.button("Predict")
    if result:
        if predict_value == 1:
            st.subheader('Sorry I am sad to say that you are diagnosed with diabetes :pensive:')
        else:
            st.subheader('I am very happy to say that you are not diagnosed with diabetes! :smile:')
