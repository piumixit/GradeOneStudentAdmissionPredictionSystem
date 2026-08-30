import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Grade 1 Admission Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Grade 1 Student Admission Predictor")

st.markdown(
    """
    ### Predict Next Year's Grade 1 Admissions

    This application uses historical Grade 1 admission data to
    estimate the number of new students expected next year.

    **Data period:** 2022–2024  
    **Prediction target:** Next year's student admissions
    """
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "admission.csv"


@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset `{DATA_FILE}` was not found.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    return df


df = load_data()


# ============================================================
# BASIC DATA CLEANING
# ============================================================

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Admissions"] = pd.to_numeric(df["Admissions"], errors="coerce")

df = df.dropna(
    subset=[
        "Year",
        "Province",
        "District",
        "Medium",
        "Gender",
        "Admissions"
    ]
)

df["Year"] = df["Year"].astype(int)
df["Admissions"] = df["Admissions"].astype(float)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Prediction Settings")

province = st.sidebar.selectbox(
    "Province",
    sorted(df["Province"].unique())
)

districts = sorted(
    df[df["Province"] == province]["District"].unique()
)

district = st.sidebar.selectbox(
    "District",
    districts
)

medium = st.sidebar.selectbox(
    "Medium",
    sorted(df["Medium"].unique())
)

gender = st.sidebar.selectbox(
    "Gender",
    sorted(df["Gender"].unique())
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Years",
        f"{df['Year'].min()}–{df['Year'].max()}"
    )

with col3:
    st.metric(
        "Districts",
        df["District"].nunique()
    )

with col4:
    st.metric(
        "Total Admissions",
        f"{int(df['Admissions'].sum()):,}"
    )


# ============================================================
# HISTORICAL DATA FOR SELECTED CATEGORY
# ============================================================

selected_data = df[
    (df["Province"] == province) &
    (df["District"] == district) &
    (df["Medium"] == medium) &
    (df["Gender"] == gender)
].sort_values("Year")


st.subheader("📈 Historical Admissions")

if len(selected_data) > 0:

    chart_data = selected_data[
        ["Year", "Admissions"]
    ].set_index("Year")

    st.line_chart(chart_data)

    st.dataframe(
        selected_data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No historical data exists for this combination."
    )


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

province_encoder = LabelEncoder()
district_encoder = LabelEncoder()
medium_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

df["Province_Code"] = province_encoder.fit_transform(
    df["Province"]
)

df["District_Code"] = district_encoder.fit_transform(
    df["District"]
)

df["Medium_Code"] = medium_encoder.fit_transform(
    df["Medium"]
)

df["Gender_Code"] = gender_encoder.fit_transform(
    df["Gender"]
)


# ============================================================
# TRAIN MODEL
# ============================================================

features = [
    "Year",
    "Province_Code",
    "District_Code",
    "Medium_Code",
    "Gender_Code"
]

X = df[features]
y = df["Admissions"]


@st.cache_resource
def train_model(X, y):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        min_samples_leaf=2
    )

    model.fit(X, y)

    return model


model = train_model(X, y)


# ============================================================
# PREDICTION
# ============================================================

st.subheader("🔮 Next Year Prediction")

next_year = int(df["Year"].max()) + 1


try:

    province_code = province_encoder.transform(
        [province]
    )[0]

    district_code = district_encoder.transform(
        [district]
    )[0]

    medium_code = medium_encoder.transform(
        [medium]
    )[0]

    gender_code = gender_encoder.transform(
        [gender]
    )[0]

    prediction_input = pd.DataFrame(
        [[
            next_year,
            province_code,
            district_code,
            medium_code,
            gender_code
        ]],
        columns=features
    )

    prediction = model.predict(
        prediction_input
    )[0]

    prediction = max(0, prediction)

    prediction = int(round(prediction))


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Admissions",
            f"{prediction:,}"
        )

    with col2:

        st.metric(
            "Prediction Year",
            next_year
        )


    st.success(
        f"Estimated Grade 1 admissions for "
        f"{gender} students in {district}, "
        f"{medium} medium for {next_year}: "
        f"**{prediction:,} students**"
    )


except Exception as e:

    st.error(
        f"Unable to generate prediction: {e}"
    )


# ============================================================
# NATIONAL PREDICTION
# ============================================================

st.subheader("🇱🇰 National Admission Prediction")


def generate_national_prediction():

    predictions = []

    for _, row in df[
        [
            "Province",
            "District",
            "Medium",
            "Gender"
        ]
    ].drop_duplicates().iterrows():

        try:

            p_code = province_encoder.transform(
                [row["Province"]]
            )[0]

            d_code = district_encoder.transform(
                [row["District"]]
            )[0]

            m_code = medium_encoder.transform(
                [row["Medium"]]
            )[0]

            g_code = gender_encoder.transform(
                [row["Gender"]]
            )[0]

            input_data = pd.DataFrame(
                [[
                    next_year,
                    p_code,
                    d_code,
                    m_code,
                    g_code
                ]],
                columns=features
            )

            pred = model.predict(input_data)[0]

            predictions.append(
                max(0, pred)
            )

        except Exception:
            continue

    return int(round(sum(predictions)))


national_prediction = generate_national_prediction()


st.metric(
    f"Estimated Sri Lankan Admissions - {next_year}",
    f"{national_prediction:,}"
)


# ============================================================
# PROVINCE SUMMARY
# ============================================================

st.subheader("🗺️ Predicted Admissions by Province")


province_predictions = []

for prov in sorted(df["Province"].unique()):

    total_prediction = 0

    prov_data = df[
        df["Province"] == prov
    ]

    combinations = prov_data[
        [
            "Province",
            "District",
            "Medium",
            "Gender"
        ]
    ].drop_duplicates()

    for _, row in combinations.iterrows():

        try:

            p_code = province_encoder.transform(
                [row["Province"]]
            )[0]

            d_code = district_encoder.transform(
                [row["District"]]
            )[0]

            m_code = medium_encoder.transform(
                [row["Medium"]]
            )[0]

            g_code = gender_encoder.transform(
                [row["Gender"]]
            )[0]

            input_data = pd.DataFrame(
                [[
                    next_year,
                    p_code,
                    d_code,
                    m_code,
                    g_code
                ]],
                columns=features
            )

            pred = model.predict(
                input_data
            )[0]

            total_prediction += max(0, pred)

        except Exception:
            pass

    province_predictions.append(
        {
            "Province": prov,
            "Predicted Admissions": int(
                round(total_prediction)
            )
        }
    )


province_df = pd.DataFrame(
    province_predictions
)

st.bar_chart(
    province_df.set_index("Province")
)

st.dataframe(
    province_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🤖 About the AI Model"):

    st.write(
        """
        The application uses a Random Forest Regression model.

        **Input features:**

        - Year
        - Province
        - District
        - Medium
        - Gender

        **Target variable:**

        - Grade 1 Admissions

        The categorical variables are converted into numerical
        representations using Label Encoding.

        The trained Random Forest model uses historical admission
        records to estimate the expected number of students for
        the following year.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Grade 1 Admission Prediction System | "
    " Cloud Computing for Artificial Intelligence"
)

