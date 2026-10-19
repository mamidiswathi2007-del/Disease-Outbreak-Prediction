from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EpiWatch | Disease Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "disease_data.csv"
MODEL_PATH = BASE_DIR / "models" / "disease_model.pkl"
CSS_PATH = BASE_DIR / "assets" / "style.css"


# =========================================================
# PROFESSIONAL DASHBOARD CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', Arial, sans-serif;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #f5f8ff 0%,
            #edf3ff 100%
        );
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 255, 0.95);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #102542 0%,
            #1e4168 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.06);
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 6px;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #16345c;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #16345c;
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e9f7;
        border-radius: 17px;
        padding: 23px;
        min-height: 135px;
        box-shadow: 0 8px 25px rgba(30, 64, 110, 0.07);
    }

    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #17365f;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 7px;
    }

    .result-card {
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0 20px 0;
    }

    .result-card h2 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 26px;
    }

    .result-card p {
        margin-bottom: 0;
        font-size: 16px;
    }

    .result-high {
        background: #fff1f2;
        border: 2px solid #e11d48;
        color: #9f1239;
    }

    .result-low {
        background: #ecfdf5;
        border: 2px solid #059669;
        color: #065f46;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e1e9f7;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(30, 64, 110, 0.05);
    }

    .info-card h3 {
        color: #17365f;
        margin-top: 0;
    }

    .info-card p {
        color: #64748b;
    }

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #2563eb,
            #1d4ed8
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 11px 22px;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button:hover {
        background: #1e40af;
        color: white;
        border: none;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 24px;
        border: 1px solid #e1e9f7;
        border-radius: 17px;
        box-shadow: 0 8px 25px rgba(30, 64, 110, 0.06);
    }

    .disclaimer {
        margin-top: 30px;
        padding: 17px;
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        color: #92400e;
        font-size: 13px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px 5px;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


df = load_dataset()
model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 15px 0 25px 0;">
            <div style="font-size: 30px; font-weight: 800;">
                🩺 EpiWatch
            </div>
            <div style="font-size: 13px; color: #cbd5e1;">
                Disease Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "Risk Assessment",
            "Analytics",
            "Dataset Explorer",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("### 🔬 System Status")

    if model is not None:
        st.success("ML Model Loaded")
    else:
        st.error("Model file not found")

    if df is not None:
        st.success("Dataset Loaded")
    else:
        st.error("Dataset file not found")

    st.caption("Academic demonstration dashboard")


# =========================================================
# COMMON VALIDATION
# =========================================================

required_columns = {
    "temperature",
    "rainfall",
    "cases",
    "outbreak",
}

data_ready = (
    df is not None
    and required_columns.issubset(set(df.columns))
)

if not data_ready:
    st.error(
        "Dataset is missing or required columns are unavailable. "
        "Check data/disease_data.csv."
    )
    st.stop()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">Disease Intelligence Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor dataset patterns and explore the demonstration model.'
        '</div>',
        unsafe_allow_html=True,
    )

    total_records = len(df)
    total_features = len(
        ["temperature", "rainfall", "cases"]
    )
    outbreak_records = int(
        (df["outbreak"] == 1).sum()
    )
    normal_records = int(
        (df["outbreak"] == 0).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "📁 Total Records", f"{total_records}", "Dataset rows"),
        (col2, "🧬 Input Features", f"{total_features}", "Model inputs"),
        (col3, "🔴 Class 1 Records", f"{outbreak_records}", "Dataset label count"),
        (col4, "🟢 Class 0 Records", f"{normal_records}", "Dataset label count"),
    ]

    for column, label, value, note in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        counts = (
            df["outbreak"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .rename_axis("Outbreak Class")
            .reset_index(name="Records")
        )

        fig = px.pie(
            counts,
            names="Outbreak Class",
            values="Records",
            title="Class Distribution",
            hole=0.55,
            color="Outbreak Class",
            color_discrete_map={
                0: "#10b981",
                1: "#ef4444",
            },
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#17365f",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        feature_data = pd.DataFrame(
            {
                "Feature": [
                    "Temperature",
                    "Rainfall",
                    "Reported Cases",
                ],
                "Average": [
                    df["temperature"].mean(),
                    df["rainfall"].mean(),
                    df["cases"].mean(),
                ],
            }
        )

        fig = px.bar(
            feature_data,
            x="Feature",
            y="Average",
            title="Average Feature Values",
            color="Feature",
            color_discrete_sequence=[
                "#2563eb",
                "#06b6d4",
                "#8b5cf6",
            ],
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#17365f",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Educational demonstration only.</strong>
            This project uses a small synthetic dataset.
            Charts describe this dataset and should not be interpreted
            as real-world disease surveillance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# RISK ASSESSMENT PAGE
# =========================================================

elif page == "Risk Assessment":

    st.markdown(
        '<div class="main-title">🤖 Outbreak Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter sample feature values to test the trained classifier.'
        '</div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "Trained model not found. Run train_model.py first."
        )
        st.stop()

    with st.form("prediction_form"):

        st.subheader("📍 Input Information")

        location = st.text_input(
            "Location (optional)",
            placeholder="e.g. Hyderabad",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-50.0,
                max_value=70.0,
                value=30.0,
                step=0.5,
            )

        with col2:
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=45.0,
                step=1.0,
            )

        with col3:
            cases = st.number_input(
                "Reported Cases",
                min_value=0,
                value=20,
                step=1,
            )

        submitted = st.form_submit_button(
            "🔍 Predict Outbreak Class",
            use_container_width=True,
        )

    if submitted:

        try:
            input_data = pd.DataFrame(
                [
                    {
                        "temperature": temperature,
                        "rainfall": rainfall,
                        "cases": cases,
                    }
                ]
            )

            prediction = int(model.predict(input_data)[0])

            st.markdown(
                '<div class="section-title">'
                '🧠 Machine Learning Result'
                '</div>',
                unsafe_allow_html=True,
            )

            if prediction == 1:

                st.markdown(
                    """
                    <div class="result-card result-high">
                        <h2>⚠️ Outbreak Class 1</h2>
                        <p>
                            The trained model assigned this input
                            to class 1 in the demonstration dataset.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="result-card result-low">
                        <h2>✅ Outbreak Class 0</h2>
                        <p>
                            The trained model assigned this input
                            to class 0 in the demonstration dataset.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-title">'
                '📋 Input Summary'
                '</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            summary = [
                (c1, "Location", location.strip() or "Not provided"),
                (c2, "Temperature", f"{temperature:.1f} °C"),
                (c3, "Rainfall", f"{rainfall:.1f} mm"),
                (c4, "Reported Cases", str(cases)),
            ]

            for column, label, value in summary:
                with column:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value"
                                 style="font-size: 24px;">
                                {value}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as exc:
            st.error(
                "Prediction failed. Check the model and its input features."
            )
            st.exception(exc)

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important:</strong>
            Class 0 and Class 1 are labels learned from the synthetic
            training dataset. They are not validated real-world risk
            categories and must not guide medical or public-health decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "Analytics":

    st.markdown(
        '<div class="main-title">📈 Data Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the numerical patterns in the available dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df,
            x="temperature",
            color="outbreak",
            nbins=10,
            title="Temperature Distribution by Class",
            labels={
                "temperature": "Temperature (°C)",
                "outbreak": "Outbreak Class",
            },
            color_discrete_sequence=["#2563eb", "#ef4444"],
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="rainfall",
            y="cases",
            color="outbreak",
            size="cases",
            title="Rainfall vs Reported Cases",
            labels={
                "rainfall": "Rainfall (mm)",
                "cases": "Reported Cases",
                "outbreak": "Outbreak Class",
            },
            color_discrete_sequence=["#10b981", "#ef4444"],
        )

        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        x="outbreak",
        y="temperature",
        color="outbreak",
        title="Temperature by Dataset Class",
        labels={
            "outbreak": "Outbreak Class",
            "temperature": "Temperature (°C)",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "These plots describe the supplied synthetic data. "
        "They do not establish causal relationships."
    )


# =========================================================
# DATASET EXPLORER PAGE
# =========================================================

elif page == "Dataset Explorer":

    st.markdown(
        '<div class="main-title">📂 Dataset Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Inspect the records loaded from disease_data.csv.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Information")
        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])
        st.write("Duplicate rows:", int(df.duplicated().sum()))

    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().rename("Missing Values")
        st.dataframe(missing, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="disease_data.csv",
        mime="text/csv",
    )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About EpiWatch</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Disease Outbreak Prediction Using Machine Learning'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Project Objective</h3>
                <p>
                    Demonstrate a basic Machine Learning classification
                    workflow using temperature, rainfall, and reported
                    case counts as input features.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>🧰 Technologies</h3>
                <p>
                    Python · Pandas · Scikit-learn · Joblib ·
                    Plotly · Streamlit
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Project Workflow</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        1. Load the CSV dataset.
        2. Inspect and prepare the input features.
        3. Train the classification model using `train_model.py`.
        4. Save the trained model in the `models` folder.
        5. Enter sample values in the dashboard.
        6. Display the class predicted by the model.
        """
    )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Dataset limitation:</strong>
            This project uses a small synthetic dataset created for
            educational demonstration. A high test score on such a small
            dataset does not establish real-world predictive performance.
            This application is not a diagnostic or public-health tool.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        EpiWatch | Disease Outbreak Prediction Project<br>
        Built for academic demonstration using Python and Machine Learning.
    </div>
    """,
    unsafe_allow_html=True,
)