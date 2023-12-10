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
set_png_as_page_bg('phone.jpg')


classifier_name=['Xgboost']
option = st.sidebar.selectbox('Модель', classifier_name)
st.subheader(option)



model=pickle.load(open('model_saved','rb'))



def predict_churn(GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption,
                 r1,r2,r3,r4,r5,r6,r7,r8,r9,r10):
    features = np.array([[GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption,
                 r1,r2,r3,r4,r5,r6,r7,r8,r9,r10]])
    features = pd.DataFrame(features)
    features = features.apply(pd.to_numeric, errors="ignore")
    features.columns = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 
                          'Australia, New Zealand and Northern America',	'CSE Asia',	'Eastern Asia',	'Eastern Europe',	'Latin America and the Caribbean',
                          'Northern Africa', 'Northern and Western Europe',	'Southern Europe',	'Sub-Saharan Africa',	'Western Asia']                     
    input = features
    prediction = model.predict(input)
    return float(prediction)    


def main():
    st.title("Happiness level")
    html_temp = """
    <div style="background-color:white ;padding:5px">
    <h2 style="color:black;text-align:center;">Fill out this form</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    avg = st.selectbox('Average coefficients by region: if you do not know the value of the coefficient, you can indicate the average value for the region.',
                               ['Select coefficient', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices','Generosity', 'Perceptions of corruption'])
    if avg == 'GDP per capita':
        st.image('1.png')
    if avg == 'Social support':
        st.image('2.png')
    if avg == 'Healthy life expectancy':
        st.image('3.png')
    if avg == 'Freedom to make life choices':
        st.image('4.png')
    if avg == 'Generosity':
        st.image('5.png')
    if avg == 'Perceptions of corruption':
        st.image('6.png')


    st.sidebar.image('prior.png', width=300)
    st.sidebar.subheader("A model for predicting the level of happiness.")
    st.sidebar.subheader("You must enter odds data. If you do not know the coefficient, you can look at the average levels of the indicator for the region and set a similar value.")
    st.sidebar.text("Developer - Aleksey Dubovcov")


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
        r3 = 1
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

    
    Happy_html = """  
              <div style="background-color:#bcf799;padding:20px >
               <h2 style="color:green;text-align:center;"> This is a happy place. <br>You can see indicators that affect the level of happiness in your region.</h2>
               </div>
            """
    
    Normal_html = """  
              <div style="background-color: #aefcf5;padding:20px >
               <h2 style="color:green;text-align:center;"> The level of happiness in this place is normal. <br>You can see indicators that affect the level of happiness in your region.</h2>
               </div>
            """
    
    No_happy_html = """  
              <div style="background-color:#ffbdbd;padding:20px >
              <h2 style="color:red ;text-align:center;"> This is not a happy place <br>You can see indicators that affect the level of happiness in your region.</h2>
              </div>
            """

    if GDP_per_capita == 0 or Social_support == 0 or Healthy_life_expectancy  == 0 or  Freedom_to_make_life_choices  == 0 or  Generosity  == 0 or  Perceptions_of_corruption == 0:
        st.error('Enter values for each coefficient')
    
    else: 
        st.button('Predict'):      
            output = predict_churn(GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption,
                 r1,r2,r3,r4,r5,r6,r7,r8,r9,r10)
       

                if output >= 7:
                    st.success('Happiness score {:.2f}'.format(output))
                    st.markdown(Happy_html, unsafe_allow_html= True)
                    st.balloons()
                elif output < 5.5:
                    st.error('Happiness score {:.2f}'.format(output))
                    st.markdown(No_happy_html, unsafe_allow_html= True)
                else:
                    st.success('Happiness score {:.2f}'.format(output))
                    st.markdown(Normal_html, unsafe_allow_html= True)
                    st.balloons()


if __name__=='__main__':
    main()

