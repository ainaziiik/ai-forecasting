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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Student Dashboard",
    layout="wide"
)

# =========================================================
# MODERN UI
# =========================================================

st.markdown("""
<style>

/* =======================================================
BACKGROUND
======================================================= */

.stApp {

    background:
        radial-gradient(circle at top left, #7c3aed 0%, transparent 25%),
        radial-gradient(circle at bottom right, #a855f7 0%, transparent 25%),
        linear-gradient(135deg, #0f0f1a, #151528, #1e1b4b);

    color: white;
}

/* =======================================================
GLOBAL TEXT
======================================================= */

html, body, [class*="css"]  {

    color: #f5f3ff !important;
}

/* =======================================================
SIDEBAR
======================================================= */

section[data-testid="stSidebar"] {

    background: rgba(17, 24, 39, 0.75);

    backdrop-filter: blur(20px);

    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar text */

section[data-testid="stSidebar"] * {

    color: #a398b5 !important;
}

/* =======================================================
HEADINGS
======================================================= */

h1 {

    color: white !important;

    font-size: 52px !important;

    font-weight: 800 !important;
}

h2, h3 {

    color: #ffffff !important;
}

/* =======================================================
HERO CARD
======================================================= */

.hero {

    background: rgba(255,255,255,0.06);

    border-radius: 30px;

    padding: 40px;

    backdrop-filter: blur(20px);

    border: 1px solid rgba(255,255,255,0.08);

    margin-bottom: 25px;
}

/* =======================================================
METRICS
======================================================= */

div[data-testid="metric-container"] {

    background: rgba(255,255,255,0.07);

    border: 1px solid rgba(255,255,255,0.08);

    padding: 20px;

    border-radius: 24px;

    backdrop-filter: blur(18px);

    box-shadow:
        0 8px 30px rgba(124,58,237,0.25);
}

/* Metric text */

div[data-testid="metric-container"] * {

    color: white !important;
}

/* =======================================================
TABS
======================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap: 14px;
}

.stTabs [data-baseweb="tab"] {

    background: rgba(255,255,255,0.06);

    border-radius: 16px;

    padding: 12px 24px;

    color: white !important;

    border: 1px solid rgba(255,255,255,0.08);

    transition: 0.3s;
}

.stTabs [aria-selected="true"] {

    background: linear-gradient(
        135deg,
        #7c3aed,
        #a855f7
    ) !important;

    color: white !important;
}

/* =======================================================
DATAFRAME
======================================================= */

[data-testid="stDataFrame"] {

    border-radius: 20px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.08);
}

/* =======================================================
BUTTONS
======================================================= */

.stButton>button {

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

.stButton>button:hover {

    transform: scale(1.03);

    box-shadow:
        0 10px 30px rgba(168,85,247,0.45);
}

/* =======================================================
UPLOAD
======================================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.06);

    border-radius: 22px;

    padding: 18px;

    border: 1px dashed rgba(192,132,252,0.5);
}

/* upload text */

[data-testid="stFileUploader"] * {

    color: white !important;
}

/* =======================================================
SLIDERS
======================================================= */

/* labels */

.stSlider label {

    color: white !important;

    font-size: 16px !important;
}

/* numbers */

.stSlider div {

    color: white !important;
}

/* =======================================================
INPUTS
======================================================= */

input {

    color: white !important;
}

/* =======================================================
SUCCESS / ALERTS
======================================================= */

.stSuccess,
.stInfo,
.stWarning,
.stError {

    border-radius: 18px !important;

    color: white !important;
}

/* =======================================================
SCROLLBAR
======================================================= */

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

/* =======================================================
FOOTER
======================================================= */

footer {

    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div style="
background: rgba(255,255,255,0.06);
padding:40px;
border-radius:30px;
backdrop-filter: blur(20px);
border:1px solid rgba(255,255,255,0.08);
margin-bottom:25px;
">

<h2 style="
font-size:36px;
margin-bottom:10px;
">
🧠 AI Student Performance Dashboard
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
# KYRGYZ NAMES
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
    'failures': 'Ийгиликсиздик',
    'goout': 'Сыртка чыгуу',
    'freetime': 'Бош убакыт',
    'health': 'Ден соолук',
    'traveltime': 'Жолго кеткен убакыт'
}

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Башкаруу панели")

uploaded_file = st.sidebar.file_uploader(
    "CSV файлын жүктөңүз",
    type="csv"
)

# =========================================================
# MAIN
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file, sep=';')

        if df.empty:
            st.error("CSV файлы бош.")
            st.stop()

        # =====================================================
        # KPI
        # =====================================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "👨‍🎓 Окуучулар",
            len(df)
        )

        col2.metric(
            "📊 Орточо баа",
            round(df['G3'].mean(), 2)
        )

        col3.metric(
            "📚 Колонкалар",
            len(df.columns)
        )

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3 = st.tabs([
            "📊 Анализ",
            "🤖 AI Моделдер",
            "🔮 Прогноз"
        ])

        # =====================================================
        # TAB 1
        # =====================================================

        with tab1:

            st.subheader("📋 Маалыматтар таблицасы")

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

            # -------------------------------------------------

            st.subheader("🔥 Факторлордун байланышы")

            numeric_df = df.select_dtypes(include=[np.number])

            fig_corr, ax_corr = plt.subplots(figsize=(12, 6))

            sns.heatmap(
                numeric_df.corr(),
                cmap='magma',
                ax=ax_corr
            )

            st.pyplot(fig_corr)

            # -------------------------------------------------

            st.subheader("📈 Баалардын бөлүштүрүлүшү")

            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))

            sns.histplot(
                df['G3'],
                kde=True,
                ax=ax_hist
            )

            st.pyplot(fig_hist)

        # =====================================================
        # DATA PREP
        # =====================================================

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

        drop_cols = [
            col for col in ['G1', 'G2', 'G3', 'target']
            if col in data_clean.columns
        ]

        X = data_clean.drop(drop_cols, axis=1)

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
        # MODELS
        # =====================================================

        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }

        results = {}

        best_model = None
        best_accuracy = 0
        best_predictions = None
        best_model_name = ""

        with st.spinner("🤖 AI модель окутулууда..."):

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
        # TAB 2
        # =====================================================

        with tab2:

            st.subheader("🏆 Моделдерди салыштыруу")

            results_df = pd.DataFrame({
                "Модель": results.keys(),
                "Тактык": [
                    f"{acc:.2%}"
                    for acc in results.values()
                ]
            })

            st.dataframe(results_df)

            # -------------------------------------------------

            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))

            pd.Series(results).sort_values().plot(
                kind='barh',
                ax=ax_bar
            )

            ax_bar.set_xlabel("Тактык")

            st.pyplot(fig_bar)

            # -------------------------------------------------

            st.success(
                f"✨ Эң жакшы модель: "
                f"{best_model_name} "
                f"({best_accuracy:.2%})"
            )

            # -------------------------------------------------

            if hasattr(best_model, 'feature_importances_'):

                st.subheader("📌 Маанилүү факторлор")

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

            # -------------------------------------------------

            st.subheader("🎯 Confusion Matrix")

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

            ax_cm.set_xlabel("Божомол")
            ax_cm.set_ylabel("Чыныгы жооп")

            st.pyplot(fig_cm)

            # -------------------------------------------------

            st.download_button(
                "📥 Натыйжаларды жүктөө",
                results_df.to_csv(index=False),
                file_name="ai_results.csv"
            )

        # =====================================================
        # TAB 3
        # =====================================================

        with tab3:

            st.subheader("🔮 Окуучунун жыйынтыгын болжолдоо")

            age = st.slider("Жашы", 15, 22, 17)

            studytime = st.slider(
                "Окуу убактысы",
                1,
                4,
                2
            )

            absences = st.slider(
                "Сабак калтыруу",
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

            if st.button("🤖 Прогноз жасоо"):

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
                        "🎉 Окуучу ийгиликтүү өтүшү мүмкүн!"
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
