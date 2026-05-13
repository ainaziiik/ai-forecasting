import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------
# PAGE
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Окуучу Прогнозу",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Основной фон */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1e293b
    );
    border-right: 1px solid #334155;
}

/* Заголовок */
h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 42px !important;
    font-weight: 800;
    margin-bottom: 10px;
}

/* Подзаголовки */
h2, h3 {
    color: #f8fafc;
}

/* Карточки */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1e293b;
    border-radius: 12px;
    color: white;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background-color: #0ea5e9 !important;
    color: white !important;
}

/* Таблицы */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid #334155;
}

/* Кнопки */
.stButton>button {
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #2563eb
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(14,165,233,0.5);
}

/* Upload */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    border: 1px dashed #38bdf8;
}

/* Графики */
.element-container img {
    border-radius: 15px;
}

/* Footer */
footer {
    visibility: hidden;
}

/* Скролл */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #38bdf8;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("Жасалма интеллект технологияларын колдонуу менен студенттердин жетишүүсүн болжолдуу аныктоо (прогноздоо)")
st.markdown("""
<div style="
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.1);
    margin-bottom:20px;
">
    <h2 style="color:#38bdf8;">
        🤖 Жасалма интеллект аркылуу окуучулардын жетишкендигин анализдөө
    </h2>
    <p style="font-size:18px; color:#cbd5e1;">
        Machine Learning • Data Science • Streamlit Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# KYRGYZ COLUMN NAMES
# ---------------------------------------------------

kyrgyz_columns = {
    'age': 'Жашы',
    'sex': 'Жынысы',
    'studytime': 'Окуу убактысы',
    'absences': 'Сабак калтыруу',
    'G3': 'Жыйынтык баа',
    'Medu': 'Апасынын билими',
    'Fedu': 'Атасынын билими',
    'Mjob': 'Апасынын жумушу',
    'Fjob': 'Атасынын жумушу',
    'failures': 'Ийгиликсиздик',
    'goout': 'Сыртка чыгуу',
    'freetime': 'Бош убакыт',
    'health': 'Ден соолук',
    'traveltime': 'Жолго кеткен убакыт'
}

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Параметрлер")

uploaded_file = st.sidebar.file_uploader(
    "student-mat.csv файлын жүктөңүз",
    type="csv"
)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, sep=';')

        if df.empty:
            st.error("CSV файлы бош.")
            st.stop()

        if 'G3' not in df.columns:
            st.error("G3 колонкасы табылган жок.")
            st.stop()

        # ---------------------------------------------------
        # TABS
        # ---------------------------------------------------

        tab1, tab2 = st.tabs([
            "Маалыматтарды анализдөө",
            "ЖИ Моделдери"
        ])

        # ===================================================
        # TAB 1
        # ===================================================

        with tab1:

            st.subheader("Маалыматтардын таблицасы")

            preview_cols = [
                col for col in
                ['age', 'sex', 'studytime', 'absences', 'G3']
                if col in df.columns
            ]

            preview_df = df[preview_cols].copy()

            preview_df.rename(
                columns=kyrgyz_columns,
                inplace=True
            )

            st.dataframe(preview_df.head(10))

            # ---------------- HEATMAP ----------------

            st.subheader("Факторлордун байланышы")

            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.shape[1] > 1:

                fig_corr, ax_corr = plt.subplots(figsize=(12, 6))

                sns.heatmap(
                    numeric_df.corr(),
                    cmap='coolwarm',
                    ax=ax_corr
                )

                st.pyplot(fig_corr)

        # ===================================================
        # TAB 2
        # ===================================================

        with tab2:

            st.subheader("Жасалма интеллект моделдерин салыштыруу")

            # ---------------- CLEAN DATA ----------------

            data_clean = df.copy()

            for col in data_clean.columns:

                if data_clean[col].dtype == 'object':

                    le = LabelEncoder()

                    data_clean[col] = le.fit_transform(
                        data_clean[col].astype(str)
                    )

            data_clean = data_clean.fillna(0)

            # ---------------- TARGET ----------------

            data_clean['target'] = data_clean['G3'].apply(
                lambda x: 1 if x >= 10 else 0
            )

            # ---------------- FEATURES ----------------

            drop_cols = [
                col for col in ['G1', 'G2', 'G3', 'target']
                if col in data_clean.columns
            ]

            X = data_clean.drop(drop_cols, axis=1)
            y = data_clean['target']

            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(0)

            # ---------------- SPLIT ----------------

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # ===================================================
            # MODELS
            # ===================================================

            models = {
                "Логистикалык регрессия": LogisticRegression(max_iter=1000),
                "Кокустук токой (Random Forest)": RandomForestClassifier(random_state=42),
                "Чечим дарагы (Decision Tree)": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB()
            }

            results = {}

            best_model = None
            best_accuracy = 0
            best_predictions = None

            # ---------------- TRAIN MODELS ----------------

            for name, model in models.items():

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                results[name] = accuracy

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_predictions = y_pred
                    best_model_name = name

            # ===================================================
            # RESULTS TABLE
            # ===================================================

            st.write("### Моделдердин тактыгы")

            results_df = pd.DataFrame({
                "Модель": results.keys(),
                "Тактык": [
                    f"{acc:.2%}"
                    for acc in results.values()
                ]
            })

            st.dataframe(results_df)

            # ===================================================
            # BAR CHART
            # ===================================================

            st.write("### Моделдерди салыштыруу")

            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))

            pd.Series(results).sort_values().plot(
                kind='barh',
                ax=ax_bar
            )

            ax_bar.set_xlabel("Тактык")

            st.pyplot(fig_bar)

            # ===================================================
            # BEST MODEL
            # ===================================================

            st.success(
                f"Эң жакшы модель: {best_model_name} "
                f"({best_accuracy:.2%})"
            )

            # ===================================================
            # FEATURE IMPORTANCE
            # ===================================================

            if hasattr(best_model, 'feature_importances_'):

                st.write("### Эң маанилүү факторлор")

                importances = best_model.feature_importances_

                display_features = [
                    kyrgyz_columns.get(c, c)
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

            # ===================================================
            # CONFUSION MATRIX
            # ===================================================

            st.write("### Ката/Дал келбөө матрицасы (Confusion Matrix")

            cm = confusion_matrix(y_test, best_predictions)

            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Greens',
                ax=ax_cm
            )

            ax_cm.set_xlabel("Божомол")
            ax_cm.set_ylabel("Чыныгы жооп")

            st.pyplot(fig_cm)

    except Exception as e:

        st.error("Ката пайда болду")
        st.exception(e)

else:
    st.info("CSV файлын жүктөңүз.")
