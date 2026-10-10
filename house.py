
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


# Page settings
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Minimal styling
st.markdown("""
<style>
    .block-container {
        max-width: 700px;
        padding-top: 3rem;
    }

    h1 {
        text-align: center;
        color: #222222;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        margin-bottom: 30px;
    }

    .price {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #167d63;
        padding: 20px;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "UNDER_CONSTRUCTION", "RERA", "BHK_NO.",
        "SQUARE_FT", "READY_TO_MOVE", "RESALE",
        "LONGITUDE", "LATITUDE", "TARGET(PRICE_IN_LACS)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["BHK_NO.", "SQUARE_FT", "TARGET(PRICE_IN_LACS)"]
    )

    df = df[
        (df["SQUARE_FT"] > 0)
        & (df["BHK_NO."] > 0)
        & (df["TARGET(PRICE_IN_LACS)"] > 0)
    ]

    for col in ["ADDRESS", "POSTED_BY", "BHK_OR_RK"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


FEATURES = [
    "POSTED_BY", "UNDER_CONSTRUCTION", "RERA",
    "BHK_NO.", "BHK_OR_RK", "SQUARE_FT",
    "READY_TO_MOVE", "RESALE", "ADDRESS",
    "LONGITUDE", "LATITUDE"
]

NUMERIC = [
    "UNDER_CONSTRUCTION", "RERA", "BHK_NO.",
    "SQUARE_FT", "READY_TO_MOVE", "RESALE",
    "LONGITUDE", "LATITUDE"
]

CATEGORICAL = ["POSTED_BY", "BHK_OR_RK", "ADDRESS"]

TARGET = "TARGET(PRICE_IN_LACS)"


@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regression", Ridge(alpha=10.0))
    ])

    model.fit(X_train, y_train)
    return model


try:
    df = load_data()
    model = train_model(df)
except Exception as error:
    st.error(f"Could not load the dataset or train the model: {error}")
    st.stop()


# Website heading
st.title("🏠 House Price Predictor")

st.markdown(
    '<p class="subtitle">Estimate a property price in India</p>',
    unsafe_allow_html=True
)

st.divider()


# User inputs
addresses = df["ADDRESS"].value_counts().head(200).index.tolist()

address = st.selectbox("Location", addresses)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        max_value=50000,
        value=1000,
        step=100
    )

with col2:
    bedrooms = st.selectbox(
        "Bedrooms (BHK)",
        list(range(1, 11)),
        index=1
    )

property_type = st.selectbox("Property type", ["BHK", "RK"])

posted_by = st.selectbox(
    "Listed by",
    sorted(df["POSTED_BY"].unique().tolist())
)

col3, col4 = st.columns(2)

with col3:
    under_construction = st.selectbox(
        "Under construction?",
        ["No", "Yes"]
    )

with col4:
    rera = st.selectbox("RERA registered?", ["Yes", "No"])

resale = st.selectbox("Property for resale?", ["Yes", "No"])


# Predict
if st.button("Predict Price", use_container_width=True):

    area_per_bedroom = area / bedrooms

    if area_per_bedroom > 2500 or area_per_bedroom < 250:
        st.warning(
            "Please check the area and bedroom count. "
            "This combination may be unusual."
        )

    location_data = df[df["ADDRESS"] == address]

    new_property = pd.DataFrame([{
        "POSTED_BY": posted_by,
        "UNDER_CONSTRUCTION": int(under_construction == "Yes"),
        "RERA": int(rera == "Yes"),
        "BHK_NO.": bedrooms,
        "BHK_OR_RK": property_type,
        "SQUARE_FT": area,
        "READY_TO_MOVE": int(under_construction == "No"),
        "RESALE": int(resale == "Yes"),
        "ADDRESS": address,
        "LONGITUDE": location_data["LONGITUDE"].median(),
        "LATITUDE": location_data["LATITUDE"].median()
    }])

    new_property = new_property.fillna(
        df[FEATURES].median(numeric_only=True).to_dict()
    )

    price_lakhs = max(0, float(model.predict(new_property)[0]))
    price_rupees = price_lakhs * 100000

    st.divider()
    st.markdown(
        f'<div class="price">₹{price_rupees:,.0f}'
        f'<br><span style="font-size:15px;font-weight:normal;">'
        f'{price_lakhs:,.2f} lakhs estimated</span></div>',
        unsafe_allow_html=True
    )

    st.caption(
        "This is an estimate based on historical property listings, "
        "not a guaranteed selling price."
    )
