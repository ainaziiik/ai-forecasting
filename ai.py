import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="AI Окуучу Прогнозу",
    layout="wide"
)

st.title("🚀 Машиналык окутуу: Окуучулардын жетишкендигин болжолдоо")
st.markdown("---")

# ---------------- KYRGYZ NAMES ----------------

kyrgyz_names = {
    'age': 'Жашы',
    'sex': 'Жынысы',
    'Medu': 'Апасынын билими',
    'Fedu': 'Атасынын билими',
    'Mjob': 'Апасынын жумушу',
    'Fjob': 'Атасынын жумушу',
    'studytime': 'Окууга жумшаган убактысы',
    'failures': 'Ийгиликсиздиктери',
    'absences': 'Калтыруулары',
    'goout': 'Достору менен чыгуу',
    'freetime': 'Бош убактысы',
    'health': 'Ден соолугу',
    'traveltime': 'Жолго кеткен убакыт'
}

# ---------------- SIDEBAR ----------------

st.sidebar.header("Параметрлер")

uploaded_file = st.sidebar.file_uploader(
    "student-mat.csv файлын жүктөңүз",
    type="csv"
)

# ---------------- MAIN ----------------

if uploaded_file is not None:

    try:
        # CSV окуу
        df = pd.read_csv(uploaded_file, sep=';')

        # Бош файл текшерүү
        if df.empty:
            st.error("CSV файлы бош.")
            st.stop()

        # G3 текшерүү
        if 'G3' not in df.columns:
            st.error("CSV файлында 'G3' колонкасы жок.")
            st.stop()

        tab1, tab2 = st.tabs([
            "📊 Маалыматтарды анализдөө",
            "🤖 ЖИ Модели"
        ])

        # ==================================================
        # TAB 1
        # ==================================================

        with tab1:

            st.subheader("Маалыматтардын фрагменти")

            preview_cols = [
                col for col in
                ['age', 'sex', 'studytime', 'absences', 'G3']
                if col in df.columns
            ]

            st.dataframe(df[preview_cols].head(10))

            st.subheader("Факторлордун өз ара байланышы (Heatmap)")

            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.shape[1] > 1:

                fig_corr, ax_corr = plt.subplots(figsize=(12, 6))

                sns.heatmap(
                    numeric_df.corr(),
                    cmap='coolwarm',
                    ax=ax_corr
                )

                st.pyplot(fig_corr)

            else:
                st.warning("Heatmap үчүн сандык маалымат жетишсиз.")

        # ==================================================
        # TAB 2
        # ==================================================

        with tab2:

            st.subheader("Random Forest модели")

            # Көчүрмө алуу
            data_clean = df.copy()

            # ---------- TEXT -> NUMBER ----------

            for col in data_clean.columns:

                if data_clean[col].dtype == 'object':

                    le = LabelEncoder()

                    data_clean[col] = le.fit_transform(
                        data_clean[col].astype(str)
                    )

            # ---------- NaN CLEAN ----------

            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.fillna(0)

            # ---------- TARGET ----------

            data_clean['target'] = data_clean['G3'].apply(
                lambda x: 1 if x >= 10 else 0
            )

            # ---------- FEATURES ----------

            drop_cols = [
                col for col in ['G1', 'G2', 'G3', 'target']
                if col in data_clean.columns
            ]

            X = data_clean.drop(drop_cols, axis=1)

            y = data_clean['target']

            # ---------- TO NUMERIC ----------

            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(0)

            # ---------- TRAIN TEST ----------

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # ---------- MODEL ----------

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            # ---------- PREDICT ----------

            y_pred = model.predict(X_test)

            # ==================================================
            # RESULTS
            # ==================================================

            col_acc, col_cm = st.columns([1, 1])

            # ---------- ACCURACY ----------

            with col_acc:

                accuracy = accuracy_score(y_test, y_pred)

                st.metric(
                    "Random Forest тактыгы",
                    f"{accuracy:.2%}"
                )

                st.write("#### Ийгиликтин негизги факторлору")

                importances = model.feature_importances_

                display_features = [
                    kyrgyz_names.get(c, c)
                    for c in X.columns
                ]

                feat_imp = pd.Series(
                    importances,
                    index=display_features
                )

                fig_f, ax_f = plt.subplots(figsize=(8, 5))

                feat_imp.nlargest(10).sort_values().plot(
                    kind='barh',
                    ax=ax_f
                )

                st.pyplot(fig_f)

            # ---------- CONFUSION MATRIX ----------

            with col_cm:

                st.write("#### Болжолдоонун тактыгы")

                cm = confusion_matrix(y_test, y_pred)

                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Greens',
                    ax=ax_cm
                )

                ax_cm.set_xlabel("Болжолдоо")
                ax_cm.set_ylabel("Чындык")

                st.pyplot(fig_cm)

    except Exception as e:

        st.error("Ката пайда болду:")
        st.exception(e)

else:
    st.info("Сураныч, сол тараптан CSV файлын жүктөңүз.")
