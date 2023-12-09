#import sklearn
#from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
import pickle
import numpy as np
import pandas as pd

import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
     background-image: url("data:image/png;base64,%s");
     background-size: cover;
     }
     </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('6.jpg')


classifier_name=['Xgboost']
option = st.sidebar.selectbox('Модель', classifier_name)
st.subheader(option)



model=pickle.load(open('model_saved','rb'))



def predict_churn(GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption,
                 R1,R2,R3,R4,R5,R6,R7,R8,R9,R10):
    features = np.array([[GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption,
                 R1,R2,R3,R4,R5,R6,R7,R8,R9,R10]])
    features = pd.DataFrame(features)
    features = features.apply(pd.to_numeric, errors="ignore")
    features.columns = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 
                          'Australia, New Zealand and Northern America',	'CSE Asia',	'Eastern Asia',	'Eastern Europe',	'Latin America and the Caribbean',
                          'Northern Africa', 'Northern and Western Europe',	'Southern Europe',	'Sub-Saharan Africa',	'Western Asia']                     
    input = features
    prediction = model.predict(input)
    return float(prediction)    


def main():
    st.title("Прогноз оттока клиентов")
    html_temp = """
    <div style="background-color:white ;padding:5px">
    <h2 style="color:black;text-align:center;">Заполни форму</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.image('prior.png', width=300)
    st.sidebar.subheader("Итоговая работа в рамках курса Diving into Darkness of Data Science")
    st.sidebar.text("Разработчик - Дубовцов А.А.")


    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    r5 = 0
    r6 = 0
    r7 = 0
    r8 = 0
    r9 = 0
    r10 = 0


    GDP_per_capita = float(st.number_input('GDP per capita', min_value=0.00))
    Social_support = float(st.number_input('Social support', min_value=0.00))
    Healthy_life_expectancy = float(st.number_input('Healthy life expectancy', min_value=0.00))
    Freedom_to_make_life_choices = float(st.number_input('Freedom to make life choices', min_value=0.00))
    Generosity = float(st.number_input('Generosity', min_value=0.00))
    Perceptions_of_corruption = float(st.number_input('Perceptions of corruption', min_value=0.00))
    Region = st.selectbox('Region',  ['Australia, New Zealand and Northern America',	'CSE Asia',	'Eastern Asia',	'Eastern Europe',	'Latin America and the Caribbean',
                          'Northern Africa', 'Northern and Western Europe',	'Southern Europe',	'Sub-Saharan Africa',	'Western Asia'])
    if Region == 'Australia, New Zealand and Northern America':
        r1 = 1
    elif Region == 'CSE Asia':
        r2 = 1
    elif Region == 'Eastern Asia':
        r3' = 1
    elif Region == 'Eastern Europe':
        r4 = 1
    elif Region == 'Latin America and the Caribbean':
        r5 = 1
    elif Region == 'Northern Africa':
        r6 = 1
    elif Region == 'Northern and Western Europe':
        r7= 1
    elif Region == 'Southern Europe':
        r8= 1
    elif Region == 'Sub-Saharan Africa':
        r9 = 1
    elif Region == 'Western Asia':
        r10 = 1

    
    churn_html = """  
              <div style="background-color:#f44336;padding:20px >
               <h2 style="color:red;text-align:center;"> Клиент уходит. <br>Добавить клиента в CRM кампанию: потенциально потерянные клиенты.</h2>
               </div>
            """
    
    no_churn_html = """  
              <div style="background-color:#94be8d;padding:20px >
               <h2 style="color:green ;text-align:center;"> Клиент остаётся в банке.</h2>
               </div>
            """
    
    mb_churn_html = """  
              <div style="background-color:#c9c7c7;padding:20px >
              <h2 style="color:blue ;text-align:center;"> Клиент может уйти из банка. <br>Добавить клиента в CRM кампанию: удержание клиентов.</h2>
              </div>
            """

    
    if st.button('Сделать прогноз'):      
        output = predict_churn(GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption,
                 R1,R2,R3,R4,R5,R6,R7,R8,R9,R10)
        st.balloons()

        if output >= 6:
            st.markdown(churn_html, unsafe_allow_html= True)
            st.success('уровень счастья {:.2f}'.format(output))

        else:
            st.markdown(no_churn_html, unsafe_allow_html= True)    
            st.success('уровень счастья {:.2f}'.format(output))


if __name__=='__main__':
    main()

