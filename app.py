import sklearn
#from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
import pickle
import numpy as np

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



model=pickle.load(open("model_saved","rb"))



def predict_churn(Region, GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption):
    input = np.array([[Region, GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption]])
    prediction = model.predict_proba(input)[:, 1]
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

    Region = st.selectbox('Region', ['Australia, New Zealand and Northern America', 'CSE Asia', 'Eastern Asia', 'Latin America and the Caribbean', 'Northern Africa', 'Northern and Western Europe', 'Southern Europe', 'Sub-Saharan Africa', 'Western Asia'])
    GDP_per_capita = st.number_input('GDP per capita', min_value=0.00)
    Social_support = st.number_input('Social support', min_value=0.00)
    Healthy_life_expectancy = st.number_input('Healthy life expectancy', min_value=0.00)
    Freedom_to_make_life_choices = st.number_input('Freedom to make life choices', min_value=0.00)
    Generosity = st.number_input('Generosity', min_value=0.00)
    Perceptions_of_corruption = st.number_input('Perceptions of corruption', min_value=0.00)


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

    
    st.button('Сделать прогноз'):
    output = predict_churn(Region, GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption)
                st.success('Уровень счастья составляет {:.2f} %'.format(output))
                


if __name__=='__main__':
    main()

