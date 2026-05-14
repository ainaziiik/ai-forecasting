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

# =========================================================
# Башкы бети
# =========================================================

st.set_page_config(
    page_title="Жасалма интеллект технологияларын колдонуу менен студенттердин жетишүүсүн болжолдуу аныктоо",
    layout="wide"
)

# =========================================================
# CSS стилдери
# =========================================================

st.markdown("""
<style>

/* Башкы фон */
.stApp {
    background:
        radial-gradient(circle at top left, #7c3aed 0%, transparent 25%),
        radial-gradient(circle at bottom right, #a855f7 0%, transparent 25%),
        linear-gradient(135deg, #0f0f1a, #151528, #1e1b4b);
    color: white;
}



/* Окуучулар, Орточо баа, Колонкалар */
[data-testid="stMetricLabel"] p {
    color: white !important;
    font-size: 16px !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}



/* Баскычтар */
.stButton > button, [data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #7c3aed, #c084fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 12px 26px !important;
    font-size: 15px !important;
    font-weight: 600 !important;


    
    /* Переход */
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.35) !important;
}



/* Хувер */
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
    transform: translateY(-3px) scale(1.05) !important; 
    background: linear-gradient(135deg, #a855f7, #d8b4fe) !important; 
    box-shadow: 0 10px 30px rgba(168,85,247,0.5) !important; 
    cursor: pointer !important;
}



/* Баскандагы эффект */
.stButton > button:active, [data-testid="stDownloadButton"] button:active {
    transform: translateY(1px) scale(0.98) !important;
}



/* Негизги фон */
.stApp {
    background:
        radial-gradient(circle at top left, #7c3aed 0%, transparent 25%),
        radial-gradient(circle at bottom right, #a855f7 0%, transparent 25%),
        linear-gradient(135deg, #0f0f1a, #151528, #1e1b4b);
    color: white;
}



/* Бардык текст */
html,
body,
[class*="css"] {
    color: white !important;
}



/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.82);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
    padding-top: 20px;
}



/* Sidebar тексти */
section[data-testid="stSidebar"] * {
    color: white !important;
}



/* Башкаруу панели */
section[data-testid="stSidebar"] h1 {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
    font-size: 38px !important;
}



/* Аталыш (заголовок) */
h1 {
    color: white !important;
    font-size: 52px !important;
    font-weight: 800 !important;
}

h2,
h3 {
    color: white !important;
}



/* Негизги блок */
.hero {
    background: rgba(255,255,255,0.06);
    border-radius: 30px;
    padding: 40px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 25px;
}



/* KPI карточкалары */
div[data-testid="metric-container"] {
    background: rgba(15,15,26,0.72) !important;
    border: 1px solid rgba(255,255,255,0.12);
    padding: 22px;
    border-radius: 24px;
    backdrop-filter: blur(18px);
    box-shadow:
        0 8px 30px rgba(124,58,237,0.28);
}



/* KPI тексти */
div[data-testid="metric-container"] * {
    color: white !important;
}



/* KPI цифралары */
div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}



/* БАШКЫ МААЛЫМАТТАР */
.stTabs [data-baseweb="tab-list"] {
    gap: 14px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 12px 24px;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.08);  
    transition: all 0.3s ease-in-out !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.15) !important; /* Становится светлее */
    transform: translateY(-2px) !important; /* Слегка приподнимается */
    border-color: rgba(168, 85, 247, 0.5) !important; /* Появляется фиолетовое свечение у бордера */
    cursor: pointer !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #a855f7
    ) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(0px) !important; /* Выбранная вкладка не «прыгает» */
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #a855f7 !important; /* Делаем линию под цвет бренда */
}



/* Таблица*/
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}


/* Баскычтар */
.stButton > button {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #c084fc
    );
    color: white !important;
    border: none;
    border-radius: 18px;
    padding: 12px 26px;
    font-size: 15px;
    font-weight: 600;
    transition: 0.3s;
    box-shadow:
        0 6px 20px rgba(124,58,237,0.35);
}

.stButton > button {
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow:
        0 10px 30px rgba(168,85,247,0.45);
}

/* Жүктөө баскычы */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #c084fc
    ) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
}

/*Жүктөө */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 18px;
    border: 1px dashed rgba(192,132,252,0.65);
    margin-top: 10px;
}

[data-testid="stFileUploader"] * {
    color: white !important;
}



/* Слайдер */
.stSlider label {
    color: white !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

.stSlider div {

    color: white !important;
}

/* Ийгиликтүү же Ийгиликтүү эмеч*/

.stSuccess,
.stInfo,
.stWarning,
.stError {
    border-radius: 18px !important;
    color: white !important;
}


::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(
        #7c3aed,
        #c084fc
    );

    border-radius: 20px;
}

footer {
    visibility: hidden;
}

header {

    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# Башкы блок 2
# =========================================================

st.markdown("""
<div class="hero">

<h2 style="
font-size:36px;
margin-bottom:10px;
color:white;
">
ЖИ менен прогноздоо
</h2>

<p style="
font-size:18px;
color:#d8b4fe;
">
Жасалма интеллект аркылуу окуучулардын жетишкендигин анализдөө жана болжолдоо
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# Кыргызча аталыштар
# =========================================================

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
    'failures': 'Ийгиликсиздик'
}


st.sidebar.title("⚙️ Башкаруу панели")

uploaded_file = st.sidebar.file_uploader(
    "📂 CSV файлын жүктөө",
    type="csv"
)

# =========================================================
# Негизги бөлүк
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file, sep=';')
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Окуучулар",
            len(df)
        )
        col2.metric(
            "Орточо баа",
            round(df['G3'].mean(), 2)
        )
        col3.metric(
            "Колонкалар",
            len(df.columns)
        )


        tab1, tab2, tab3 = st.tabs([
            "Анализ",
            "ЖИ Моделдери",
            "Прогноз"
        ])

        # =====================================================
        # 1-ТАБ
        # =====================================================
        with tab1:

            st.subheader(" Маалыматтар таблицасы")

            preview_df = df[
                ['age', 'sex', 'studytime', 'absences', 'G3']
            ].copy()

            preview_df.rename(
                columns=kyrgyz_columns,
                inplace=True
            )

            st.dataframe(preview_df.head(10))

      
            st.subheader("Факторлордун байланышы")

            numeric_df = df.select_dtypes(include=[np.number])

            fig_corr, ax_corr = plt.subplots(figsize=(12, 6))

            sns.heatmap(
                numeric_df.corr(),
                cmap='magma',
                ax=ax_corr
            )

            st.pyplot(fig_corr)


            st.subheader(" Баалардын бөлүштүрүлүшү")

            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))

            sns.histplot(
                df['G3'],
                kde=True,
                ax=ax_hist
            )

            st.pyplot(fig_hist)


        data_clean = df.copy()

        for col in data_clean.columns:

            if data_clean[col].dtype == 'object':

                le = LabelEncoder()

                data_clean[col] = le.fit_transform(
                    data_clean[col].astype(str)
                )

        data_clean = data_clean.fillna(0)

        data_clean['target'] = data_clean['G3'].apply(
            lambda x: 1 if x >= 10 else 0
        )

        X = data_clean.drop(
            ['G1', 'G2', 'G3', 'target'],
            axis=1,
            errors='ignore'
        )

        y = data_clean['target']

        X = X.apply(pd.to_numeric, errors='coerce')

        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # =====================================================
        # Моделдер
        # =====================================================

        models = {
            "Логистикалык регрессия": LogisticRegression(max_iter=1000),
            "Кокустук токой (Random Forest)": RandomForestClassifier(random_state=42),
            "Чечим дарагы (Decision Tree)": DecisionTreeClassifier(random_state=42),
            "K-Жакын кошуналар (KNN)": KNeighborsClassifier(),
            "Ишенчээк Байес (Naive Bayes)": GaussianNB()
        }

        results = {}
        best_model = None
        best_accuracy = 0
        best_predictions = None
        best_model_name = ""
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

        
        # =====================================================
        # 2-ТАБ
        # =====================================================

        with tab2:

            st.subheader("Моделдерди салыштыруу")

            results_df = pd.DataFrame({
                "Модель": results.keys(),
                "Тактык": [
                    f"{acc:.2%}"
                    for acc in results.values()
                ]
            })

            st.dataframe(results_df)

            # =====================================================
            # Моделдердин графиги 
            # =====================================================

            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            pd.Series(results).sort_values().plot(
                kind='barh',
                ax=ax_bar
            )
            ax_bar.set_xlabel("Тактык")
            st.pyplot(fig_bar)


            st.success(
                f" Эң мыкты модель: "
                f"{best_model_name} "
                f"({best_accuracy:.2%})"
            )
=

            if hasattr(best_model, 'feature_importances_'):
                st.subheader("Маанилүү факторлор")
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


            st.subheader("Дал келүүлөр матрицасы (Confusion Matrix)")
            cm = confusion_matrix(
                y_test,
                best_predictions
            )
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Purples',
                ax=ax_cm
            )

            ax_cm.set_xlabel(""Болжолдонгон жыйынтык")
            ax_cm.set_ylabel("Чыныгы жыйынтык")

            st.pyplot(fig_cm)


            st.download_button(
                " Натыйжаларды жүктөө",
                results_df.to_csv(index=False),
                file_name="ai_results.csv"
            )
            
        # =====================================================
        # 3-ТАБ
        # =====================================================

        with tab3:

            st.subheader(" Окуучунун жыйынтыгын болжолдоо")

            age = st.slider(
                "Жашы",
                15,
                22,
                17
            )

            studytime = st.slider(
                "Окуу убактысы",
                1,
                4,
                2
            )

            absences = st.slider(
                "Сабак калтыруусу",
                0,
                30,
                5
            )

            failures = st.slider(
                "Ийгиликсиздик",
                0,
                4,
                0
            )

            if st.button("Алдын ала аныктоо!"):
                input_data = np.zeros(len(X.columns))

                feature_map = {
                    'age': age,
                    'studytime': studytime,
                    'absences': absences,
                    'failures': failures
                }

                for i, col in enumerate(X.columns):
                    if col in feature_map:
                        input_data[i] = feature_map[col]
                prediction = best_model.predict(
                    [input_data]
                )[0]
                if prediction == 1:
                    st.success(
                        "Окуучу ийгиликтүү өтүшү мүмкүн!"
                    )
                else:
                    st.error(
                        "⚠️ Кошумча даярдык керек болушу мүмкүн."
                    )

    except Exception as e:

        st.error("Ката пайда болду")

        st.exception(e)

else:

    st.info("📂 CSV файлын жүктөңүз.")
