import streamlit as st
import pickle
import numpy as np
import datetime
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler


#file data read
heart=pd.read_csv('dataset.csv')
X,y=heart,heart.target
X.drop('target',axis=1,inplace=True)
sc = StandardScaler()
X = sc.fit_transform(X)


#calucalting the age from date
def calculate_age(born):
    today = datetime.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def predict_disease_csv(csv_file):
    with open('model-pickle-RandomForest-Svc','rb') as file:
        model = pickle.load(file)
    sc2 = StandardScaler()
    csv_file = sc2.fit_transform(csv_file)
    result = model.predict(csv_file)
    return result


def predict_disease(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    
    
    if sex=="Male":
        sex=1
    else:
        sex=0
    
    
    if cp=="0 - Typical angina":
        cp=0
    elif cp=="1 - Atypical angina":
        cp=1
    elif cp=="2 - Non-anginal pain":
        cp=2
    else:
        cp=3
    
    
    if fbs=="True":
        fbs=1
    else:
        fbs=0
    

    if restecg=="0 - Normal":
        restecg=0
    elif restecg=="1 - Having ST-T wave abnormality":
        restecg=1
    else:
        restecg=2


    if exang=="Yes":
        exang=1
    else:
        exang=0

    
    if slope=="0 - Upsloping":
        slope=0
    elif slope=="1 - Flat":
        slope=1
    else:
        slope=2

    numpy_array=np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,int(ca),thal]])
    with open('model-pickle-RandomForest-Svc','rb') as file:
        model = pickle.load(file)
    result = model.predict(sc.transform(numpy_array))
    return result


def homepage():
    select=st.selectbox("Select the option to predict",("Enter details individually","Upload CSV file"))
    #Form to enter the details
    if select=="Enter details individually":
        st.write("Enter the details here")
        with st.form(key='my_form'):
            col1,col2=st.beta_columns(2)
            date_of_birth=col1.date_input("Enter date of birth",min_value=datetime.datetime(1850,1,1),max_value=datetime.datetime.today())
            sex=col1.selectbox("Select gender",("Male","Female"))
            cp=col1.selectbox("Enter chest pain",("0 - Typical angina","1 - Atypical angina","2 - Non-anginal pain","3 - Asymptomatic"))
            trestbps=col1.number_input("Enter resting blood pressure (in mm Hg on admission to the hospital)",step=1,min_value=40,max_value=200)
            chol=col1.number_input("Enter serum cholestoral in mg/dl",step=1,min_value=100,max_value=700)
            fbs=col1.selectbox("Enter fasting blood sugar &gt; 120 mg/dl",("True","False"))
            restecg=col1.selectbox("Enter resting electrocardiographic results",("0 - Normal","1 - Having ST-T wave abnormality", "2 - Probable/Definite left ventricular hypertrophy"))
            thalach=col2.number_input("Enter maximum heart rate achieved",step=1)
            exang=col2.selectbox("Enter exercise induced angina",("Yes","No"))
            oldpeak=col2.number_input("Enter ST depression induced by exercise relative to rest",step=0.5)
            slope=col2.selectbox("Enter the slope of the peak exercise ST segment",("0 - Upsloping","1 - Flat","2 - Downsloping"))
            ca=col2.selectbox("Enter the number of major vessels (0-3) colored by flourosopy",("0","1","2","3"))
            thal=col2.number_input("Enter the thalassemia value",step=1)
            col2.text("")
            col2.text("")
            submit_button = col2.form_submit_button(label='Submit')

        #onsubmit predict the result
        if submit_button:
            predicted_result=predict_disease(calculate_age(date_of_birth),sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
            if predicted_result[0]==1:
                st.error("Heart Disease has been detected")
            else:
                st.balloons()
                st.success("You are safe from heart disease")
    else:
        uploaded_file=st.file_uploader("Choose a CSV file",type="csv",help="Fields must in the sequence - name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
        try:
            if uploaded_file is not None:
                df=pd.read_csv(uploaded_file)
                name_field=df["name"]
                df=df.drop("name",axis=1)
                result=predict_disease_csv(df)
                st.subheader("Result")
                st.write(pd.concat([name_field,pd.DataFrame(result)],axis=1).rename(columns={0:'result'}))
        except:
            st.error("Error in reading the file make sure it mathces the format")
            st.markdown('''<label>For more info of format check &nbsp; </label>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle" viewBox="0 0 16 16"><path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/><path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z"/></svg>
            <label>&nbsp; on top of the file input field.</label>'''
            ,unsafe_allow_html=True)
            
    

def developers_content():
    #displaying the sata set
    st.write("The dataset that has been used for this project")
    dataset=pd.read_csv('dataset.csv')
    st.write(dataset)
    st.write("The source of this dataset is from [kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)")

    #visualization
    st.text("")
    st.subheader("Columns of dataset")
    st.text("")
    for i in range(0,len(dataset.columns),4):
        try:
            col1,col2,col3,col4,col5,col6,col7=st.beta_columns((1,0.4,1,0.4,1,0.4,1))
            col1.write(alt.Chart(pd.DataFrame(dataset[dataset.columns[i]])).mark_bar().encode(x=alt.X(dataset.columns[i], sort=None,bin=True),y='count()').properties(width=180,height=180).interactive())
            col3.write(alt.Chart(pd.DataFrame(dataset[dataset.columns[i+1]])).mark_bar().encode(x=alt.X(dataset.columns[i+1], sort=None,bin=True),y='count()').properties(width=180,height=180).interactive())
            col5.write(alt.Chart(pd.DataFrame(dataset[dataset.columns[i+2]])).mark_bar().encode(x=alt.X(dataset.columns[i+2], sort=None,bin=True),y='count()').properties(width=180,height=180).interactive())
            col7.write(alt.Chart(pd.DataFrame(dataset[dataset.columns[i+3]])).mark_bar().encode(x=alt.X(dataset.columns[i+3], sort=None,bin=True),y='count()').properties(width=180,height=180).interactive())
        except:
            pass
    
    #Heatmap
    st.subheader("Heatmap of numeric columns")
    numeric_columns=['trestbps','chol','thalach','age','oldpeak']
    # create a correlation heatmap
    fig=sns.heatmap(heart[numeric_columns].corr(), cmap="YlGnBu", annot=True, linewidths=0.1)
    fig=plt.gcf()
    fig.set_size_inches(8,6)
    st.pyplot(fig,clear_figure=True)
    st.subheader("Ensembled Algorithm (RandomForest + SVM)")
    st.code('''
    from sklearn.ensemble import VotingClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    # Training classifiers
    clf21 = RandomForestClassifier(random_state=1)
    clf23 = SVC(kernel='rbf', probability=True)

    #eclf2 = VotingClassifier(estimators=[('rf', clf21), ('knn', clf22), ('svc', clf23)],voting='hard')
    eclf2 = VotingClassifier(estimators=[('rf', clf21), ('svc', clf23)],voting='hard')

    params = {'svc__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],'rf__max_features':["log2"]}

    grid2 = GridSearchCV(estimator=eclf2, param_grid=params, cv=30)
    grid2.fit(X_train,y_train)
    grid2_predicted = grid2.predict(X_test)
    ''',language="python")
    st.write("[GitHub] (https://github.com/ChaitanyaGaneshRaju/Heart-Disease-Prediction) Link for this code")


#pageconfiguaration
st.set_page_config(layout="wide")



#style sheet


st.markdown('''
<style>
div.stButton > button{
    width:100%;
}
</style>
''',unsafe_allow_html=True)


#Tile and the subtitle
col1, col2,col3,col4,col5 = st.beta_columns((3,10,1,1,1))

image = Image.open("heart.png")
col1.image(image,use_column_width=False,width=125)
col2.title("Heart Disease Prediction")
col2.write("Using Ensembled Algorithm (RandomForest + SVM )")
st.title("Heart Disease Prediction ")
st.write("Using Ensembled Algorithm (RandomForest and SVM )")
st.markdown("***")

st.sidebar.header("Heart Disease Prediction")

homebutton=st.sidebar.button("Home")
developers_button=st.sidebar.button("For Developers")



if homebutton:
    homepage()
elif developers_button:
    developers_content()
else:
    homepage()
