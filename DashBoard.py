import streamlit as st 
import pandas as pd      
import numpy as np
import plotly.express as px 
import joblib
import catboost
import xgboost
import imblearn

# 📦 Data analysis and visualization libraries
import pandas as pd                            # For data loading and manipulation
import numpy as np                             # For numerical computations


# 🧹 Data preprocessing and cleaning
from datasist.structdata import detect_outliers        # To detect outliers in data
from sklearn.impute import SimpleImputer              # For missing value imputation (simple methods)
from sklearn.impute import KNNImputer                 # For missing value imputation (KNN-based)
from sklearn.preprocessing import RobustScaler        # For scaling features robust to outliers
from sklearn.preprocessing import OneHotEncoder       # For encoding categorical features
from sklearn.compose import ColumnTransformer         # For applying transformers to specific columns

# ⚖️ Handling imbalanced data
from imblearn.over_sampling import SMOTE              # For oversampling minority class
from imblearn.under_sampling import RandomUnderSampler  # For undersampling majority class
from imblearn.pipeline import Pipeline                # Pipeline supporting sampling steps

# 🧪 Data splitting and cross-validation
from sklearn.model_selection import train_test_split      # For splitting data into train and test sets
from sklearn.model_selection import StratifiedKFold       # For stratified K-Fold cross-validation
from sklearn.model_selection import cross_val_score, cross_val_predict  # For cross-validation

# 🧮 Classification models
from sklearn.linear_model import LogisticRegression       # Logistic Regression model
from sklearn.svm import SVC                               # Support Vector Machine model
from sklearn.neighbors import KNeighborsClassifier        # K-Nearest Neighbors model
from sklearn.tree import DecisionTreeClassifier           # Decision Tree model
from sklearn.ensemble import RandomForestClassifier       # Random Forest model
from sklearn.ensemble import GradientBoostingClassifier   # Gradient Boosting model
from sklearn.ensemble import HistGradientBoostingClassifier  # Fast version of Gradient Boosting
from xgboost import XGBClassifier                         # XGBoost model
from catboost import CatBoostClassifier                   # CatBoost model

# 🔍 Model evaluation and tuning
from sklearn.model_selection import GridSearchCV          # For hyperparameter tuning
from sklearn.metrics import accuracy_score                # For calculating accuracy
from sklearn.metrics import classification_report         # For detailed classification performance
from sklearn.metrics import confusion_matrix              # For generating confusion matrix

# 💾 Saving and loading models
import joblib  
from sklearn.ensemble import StackingClassifier
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="Diabetes prediction",
    page_icon="📊",
    layout="wide" 
)


KMeans_model = joblib.load("model_for_extract_future.pkl")
model = joblib.load("final_model.pkl")

df = pd.read_csv("diabetes_prediction_dataset.csv")

df.drop(index= df[(df["bmi"] <= 10) | (df["bmi"] >= 70)].index , inplace=True )
df.reset_index(drop=True , inplace=True)
    

smoking_risk_map = {
    "never": 0,
    "former": 1,
    "not current": 2,
    "ever": 3,
    "current": 4
}


def is_the_person_fat(bmi):
    if bmi >= 25 :
        return 1 
    else :
        return 0

def is_high_risk_age (age):
    if age >= 60 :
        return 1
    else :
        return 0

def is_high_risk_glucose(blood_glucose_level):
    if blood_glucose_level <= 140:
        return 0
    elif blood_glucose_level <= 199:
        return 1
    else :
        return 2

def hba1c_risk(val):
    if val < 5.7:
        return 0
    elif val < 6:
        return 1
    elif val < 6.5:
        return 2
    else:
        return 3

def risk_score(row):
    score = (
        row["hypertension"]
        + row["heart_disease"]
        + row["is_fat"]
        + row["smoking_risk_level"]
        + row["risk_age"]
        + row["glucose_risk_level"]
        + row["hba1c_risk_score"]
    )
    return score


st.title("Diabetes Prediction")
tabs = st.tabs(["About the Project", "Analysis", "Model Testing"])

with tabs[0]:
    st.title("📚 About the Project")

    st.markdown("""
    ## 🔍 Project Overview

    This project combines **Data Analysis** and **Machine Learning** to predict the likelihood of **Diabetes** based on health data.

    I worked with a public dataset to identify the key features influencing diabetes prediction, and then I trained a machine learning model to make predictions based on user input.

    ---

    ## 🤖 Model Performance

    - **Training Accuracy:** 97%
    - **Testing Accuracy:** 97%
    - **Validation Accuracy:** 97%

    The model performs consistently well across different data splits, showing strong generalization and reliability.

    ---

    ## 📄 Pages in This App

    ### 🟠 **Analysis Page**  
    Explore visualizations and statistical insights into the diabetes data, including feature distributions and patterns.

    ### 🟢 **Prediction Page**  
    Enter your health data (e.g., glucose, BMI, insulin levels) to see if the model predicts the likelihood of diabetes.

    > ⚠️ *Disclaimer: This tool is for educational purposes only and should not be used as a medical diagnosis.*

    ---

    ## 📝 Project Steps

    ### 1. **Data Cleaning**  
    I began with **data cleaning** to handle missing values, outliers, and inconsistencies in the dataset. This step is essential for ensuring the data is accurate and ready for analysis and modeling.

    ### 2. **Feature Engineering**  
    Next, I applied **feature engineering** to create new features that could improve the model’s performance. I also transformed the data where necessary to make it more suitable for machine learning models.

    ### 3. **Data Preprocessing**  
    During **data preprocessing**, I scaled numerical features, encoded categorical features, and split the dataset into training and testing sets.

    ### 4. **Model Training**  
    I trained several machine learning models and evaluated their performance. After testing multiple algorithms, I selected the top three performing models:
    - **XGBoost**
    - **CatBoost**
    - **HistGradientBoosting**

    These models showed promising results, so I decided to use them in a **Stacking model** to combine their strengths.

    ### 5. **Stacking Model**  
    I built a **Stacking model** that combines predictions from multiple models. This approach leverages the strengths of each individual model, improving overall prediction accuracy.

    ### 6. **Clustering for Feature Extraction**  
    To enhance the model’s performance, I used **Clustering** to extract new features. Specifically, I calculated "the distance between each point and its center or class." This technique captures additional data patterns, helping the model make more accurate predictions.

    ### 7. **Retraining the Stacking Model**  
    Finally, I retrained the **Stacking model** using the newly extracted features from the clustering step. By incorporating clustering information, the model achieved improved predictions and better overall performance.

    ---

    ### 💡 Summary:
    - Data cleaning, feature engineering, and preprocessing were performed to prepare the data.
    - Three models (XGBoost, CatBoost, HistGradientBoosting) were selected and combined into a stacking model.
    - Clustering was used to extract new features, and the model was retrained for enhanced performance.

    ---

    ## 👨‍💻 Built By:
    - **Moaaz Hassan**  
    """)

with tabs[1]:
    st.title("📊 Analysis Overview")
    st.markdown("""In this section, we explore the diabetes dataset to discover patterns, trends, and relationships between features.  
    By analyzing the data, we aim to understand which factors are most strongly associated with diabetes and how they influence the outcome.""")
    st.markdown(" ")
    st.markdown(" ")
    heart_disease_df = df.groupby(["diabetes","heart_disease"])["diabetes"].count().reset_index(name="count")
    heart_disease_df = heart_disease_df.loc[heart_disease_df["diabetes"] == 1]
    heart_disease_fig = px.pie(heart_disease_df, 
            names='heart_disease', 
            values='count', 
            title='The relationship between heart disease and diabetes',
            color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(heart_disease_fig, use_container_width=True)
    st.markdown("**14.9%** of people with **diabetes** have **Heart Disease** , if the person have Heart Disease his diabetes risk is **14.9%**")
    st.markdown(" ")

    hypertension_df = df.groupby(["diabetes","hypertension"])["diabetes"].count().reset_index(name="count")
    hypertension_df = hypertension_df.loc[hypertension_df["diabetes"] == 1]
    hypertension_fig = px.pie(hypertension_df, 
             names='hypertension', 
             values='count', 
             title='The relationship between hypertension and diabetes',
             color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(hypertension_fig, use_container_width=True)
    st.markdown("**24.6%** of people with **diabetes** have **Heart Hypertension** , if the person have Hypertension his diabetes risk is **24.6%**")
    st.markdown(" ")

    df["is_fat"] = df["bmi"].apply(is_the_person_fat)
    is_fat_df = df.groupby(["diabetes","is_fat"])["diabetes"].count().reset_index(name="count")
    is_fat_df = is_fat_df.loc[is_fat_df["diabetes"] == 1]
    is_fat_fig = px.pie(is_fat_df, 
            names='is_fat', 
            values='count', 
            title='The relationship between High weight and diabetes',
            color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(is_fat_fig, use_container_width=True)
    st.markdown("**89.1%** of people with **diabetes** are **fat** , if the person is fat his diabetes risk is **89.1%**")
    st.markdown(" ")

    df["risk_age"] = df["age"].apply(is_high_risk_age)
    # risk_age
    risk_age_df = df.groupby(["diabetes","risk_age"])["diabetes"].count().reset_index(name="count")
    risk_age_df = risk_age_df.loc[risk_age_df["diabetes"] == 1]

    risk_age_fig = px.pie(risk_age_df, 
                names='risk_age', 
                values='count', 
                title='The relationship between older age and diabetes',
                color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(risk_age_fig, use_container_width=True)
    st.markdown("**58.1%** of people with **diabetes** are **older people** , if the person is over 60 years old his diabetes risk is **41.9%**")
    st.markdown(" ")

    df["glucose_risk_level"] = df["blood_glucose_level"].apply(is_high_risk_glucose)
    # glucose_risk_level
    glucose_risk_level_df = df.groupby(["diabetes","glucose_risk_level"])["diabetes"].count().reset_index(name="count")
    glucose_risk_level_df = glucose_risk_level_df.loc[glucose_risk_level_df["diabetes"] == 1]

    glucose_risk_level_fig = px.pie(glucose_risk_level_df, 
                names='glucose_risk_level', 
                values='count', 
                title='The relationship between glucose level and diabetes',
                color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(glucose_risk_level_fig, use_container_width=True)
    st.markdown("**46.2%** of people with **diabetes** them **glucose level** is high then **199**, and **30.9%** of people with **diabetes** them **glucose level** is high then **140** , and **23%** of people with **diabetes** them **glucose level** is less than **144**")
    st.markdown(" ")

    df["hba1c_risk_score"] = df["HbA1c_level"].apply(hba1c_risk)
    # hba1c_risk_score
    hba1c_risk_score_df = df.groupby(["diabetes","hba1c_risk_score"])["diabetes"].count().reset_index(name="count")
    hba1c_risk_score_df = hba1c_risk_score_df.loc[hba1c_risk_score_df["diabetes"] == 1]
    hba1c_risk_score_fig = px.pie(hba1c_risk_score_df, 
                names='hba1c_risk_score', 
                values='count', 
                title='The relationship between hba1c score and diabetes',
                color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(hba1c_risk_score_fig, use_container_width=True)
    st.markdown("**61.1%** of people with **diabetes** them **HbA1c** scor is more then **6.5**, and **22.9%** of people with **diabetes** them **HbA1c** scor is more then **6** , and **16%** of people with **diabetes** them **HbA1c** scor is more then **5.7**")
    st.markdown(" ")


    pox_1 = px.box(data_frame=df , y="bmi" , color="diabetes" , color_discrete_sequence=px.colors.qualitative.Set2 , title="BMI Distribution by Diabetes Status")
    st.plotly_chart(pox_1, use_container_width=True)
    st.markdown("Individuals with a higher **BMI** are more likely to have diabetes compared to those with a lower BMI.")
    st.markdown(" ")

    
    pox_2 = px.box(data_frame=df , y="age" , color="diabetes" , color_discrete_sequence=px.colors.qualitative.Set2 , title="Age Distribution by Diabetes Status")
    st.plotly_chart(pox_2, use_container_width=True)
    st.markdown("Older individuals show a higher prevalence of diabetes than younger individuals.")
    st.markdown(" ")

    pox_3 = px.box(data_frame=df , y="HbA1c_level" , color="diabetes" , color_discrete_sequence=px.colors.qualitative.Set2 , title="HbA1c Level Distribution by Diabetes Status")
    st.plotly_chart(pox_3, use_container_width=True)
    st.markdown("Higher HbA1c levels are strongly associated with an increased risk of diabetes.")
    st.markdown(" ")


    pox_4 = px.box(data_frame=df , y="blood_glucose_level" , color="diabetes" , color_discrete_sequence=px.colors.qualitative.Set2 , title="Blood Glucose Level Distribution by Diabetes Status")
    st.plotly_chart(pox_4, use_container_width=True)
    st.markdown("People with elevated blood glucose levels are more frequently diagnosed with diabetes compared to those with normal levels")
    st.markdown(" ")


    


    





with tabs[2]:
    st.markdown("# **Model Prediction** 🧠")
    st.markdown("## **Please enter the values accurately for more reliable results.**")
    st.markdown(" ")

    gender = st.selectbox("Please select your gender", ["Male", "Female", "Other"])
    st.markdown(" ")

    age = st.slider("Please select your age", min_value=1, max_value=100, value=25, step=1)
    st.markdown(" ")

    hypertension = st.radio("Do you have hypertension?", ("Yes", "No"))
    hypertension_value = 1 if hypertension == "Yes" else 0
    st.markdown(" ")

    heart_disease = st.radio("Do you have heart disease?", ("Yes", "No"))
    heart_disease_value = 1 if heart_disease == "Yes" else 0
    st.markdown(" ")

    smoking_history = st.selectbox(
    "Please select your smoking history", 
    ['never', 'current', 'former', 'ever', 'not current']
    )
    st.markdown(" ")

    bmi = st.slider("Please select your bmi", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    st.markdown(" ")


    HbA1c_level = st.slider("Please select your glycated hemoglobin level", min_value=3.0, max_value=10.0, value=5.0, step=0.1)
    st.markdown(" ")

    blood_glucose_level = st.slider("Please select your blood glucose level", min_value=80, max_value=300, value=300, step=1)

    data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension_value],
        'heart_disease': [heart_disease_value],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    }

    df_values = pd.DataFrame(data)
    df_values["is_fat"] = df_values["bmi"].apply(is_the_person_fat)
    df_values["smoking_risk_level"] = df_values["smoking_history"].map(smoking_risk_map)
    df_values["risk_age"] = df_values["age"].apply(is_high_risk_age)
    df_values["glucose_risk_level"] = df_values["blood_glucose_level"].apply(is_high_risk_glucose)
    df_values["hba1c_risk_score"] = df_values["HbA1c_level"].apply(hba1c_risk)
    df_values["risk_score"] = df_values.apply(risk_score, axis=1)

    df_values[["hypertension" ,"heart_disease",
    "is_fat","smoking_risk_level"
    ,"risk_age","glucose_risk_level",
    "hba1c_risk_score","risk_score"]] = df_values[["hypertension" ,"heart_disease","is_fat",
                                            "smoking_risk_level","risk_age","glucose_risk_level",
                                            "hba1c_risk_score","risk_score"]].astype("int")
    df_values.drop(columns=["smoking_history"] , inplace=True)
    new_future = pd.DataFrame(KMeans_model.transform(df_values), columns=["distance_to_centroid_0", "distance_to_centroid_1"])
    df_values = pd.concat([df_values.reset_index(drop=True), new_future.reset_index(drop=True)], axis=1)
    df_values["age_hba1c_interaction"] = df_values["age"] * df_values["HbA1c_level"]
    df_values["bmi_glucose_interaction"] =  df_values["bmi"] * df_values["blood_glucose_level"]
    df_values["bmi_age_ratio"] = df_values["bmi"] / df_values["age"]
    df_values['glucose_triple_score'] = (df_values['blood_glucose_level'] * df_values['HbA1c_level']) / df_values['bmi']
    
    st.markdown(" ")
    if st.button('Make Prediction'):

        st.markdown("## These are your values after the feature engineering, which will be sent to the model")
        st.dataframe(df_values)
        st.markdown(" ")
        st.markdown(" ")
        prediction = model.predict(df_values)[0]
        probability = model.predict_proba(df_values)[0][1] 

        if prediction == 1:
            st.error(f"⚠️ You are likely diabetic with a probability of {probability:.2%}")
        else:
            st.success(f"✅ You are not diabetic with a probability of {probability:.2%}")

