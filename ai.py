import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="AI Окуучу Прогнозу", layout="wide")

st.title("🚀 Машиналык окутуу: Окуучулардын жетишкендигин болжолдоо")
st.markdown("---")

kyrgyz_names = {
    'age': 'Жашы', 'sex': 'Жынысы', 'Medu': 'Апасынын билими', 'Fedu': 'Атасынын билими',
    'Mjob': 'Апасынын жумушу', 'Fjob': 'Атасынын жумушу', 'studytime': 'Окууга жумшаган убактысы',
    'failures': 'Ийгиликсиздиктери', 'absences': 'Калтыруулары', 'goout': 'Достору менен чыгуу',
    'freetime': 'Бош убактысы', 'health': 'Ден соолугу', 'traveltime': 'Жолго кеткен убакыт'
}

st.sidebar.header("Параметрлер")
uploaded_file = st.sidebar.file_uploader("student-mat.csv файлын жүктөңүз", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    
    tab1, tab2 = st.tabs(["📊 Маалыматтарды анализдөө", "🤖 ЖИ Модели"])

    with tab1:
        st.subheader("Маалыматтардын фрагменти")
        st.dataframe(df[['age', 'sex', 'studytime', 'absences', 'G3']].head(10))
        
        st.subheader("Факторлордун өз ара байланышы (Heatmap)")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    with tab2:

    data = df.copy()

    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))

    data = data.fillna(0)

    data['target'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

    X = data.drop(['G3', 'target', 'G1', 'G2'], axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        col_acc, col_cm = st.columns([1, 1])
        
        with col_acc:
            st.metric("Random Forest тактыгы", f"{accuracy_score(y_test, y_pred):.2%}")
            st.write("#### Ийгиликтин негизги факторлору")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=[kyrgyz_names.get(c, c) for c in X.columns])
            fig_f, ax_f = plt.subplots()
            feat_imp.nlargest(10).sort_values().plot(kind='barh', color='skyblue', ax=ax_f)
            st.pyplot(fig_f)

        with col_cm:
            st.write("#### Болжолдоонун тактыгы (Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm)
            plt.xlabel('Болжолдоо')
            plt.ylabel('Чындык')
            st.pyplot(fig_cm)
else:
    st.warning("Көрсөтүүнү баштоо үчүн CSV файлын жүктөңүз.")
