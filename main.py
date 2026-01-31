import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# ML Libraries
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EstateAI Pro",
    page_icon="🏠",
    layout="wide"
)

# --- ANIMATION LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Animations
lottie_robot = load_lottieurl("https://lottie.host/5a882939-5b32-4115-9988-29538356779a/1LqC8XQ6S8.json")
lottie_house = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_w51pcehl.json")

# ---------------------------------------------------------
# 2. INTRO SPLASH SCREEN (FAST VERSION ⚡)
# ---------------------------------------------------------
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

if st.session_state.first_visit:
    
    intro_holder = st.empty()
    
    with intro_holder.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            if lottie_robot:
                st_lottie(lottie_robot, height=250, key="intro_robot")
            
            name_placeholder = st.empty()
            full_text = "DEVELOPED BY AJAY SAVARE"
            current_text = ""
            
            # FAST LOOP
            for char in full_text:
                current_text += char
                name_placeholder.markdown(f"""
                <h2 style='text-align: center; color: #FF512F; font-size: 45px; font-family: monospace; font-weight: bold;'>
                    {current_text}<span style='color: white; animation: blink 1s infinite;'>|</span>
                </h2>
                """, unsafe_allow_html=True)
                time.sleep(0.05) 
            
            time.sleep(0.7)
    
    intro_holder.empty()
    st.session_state.first_visit = False

# ---------------------------------------------------------
# 3. MAIN DASHBOARD LOGIC
# ---------------------------------------------------------

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark Theme */
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
    }
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 18px;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(221, 36, 118, 0.6);
    }
    /* Inputs */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# FILES
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
DATA_FILE = "housing.csv"

# --- TRAINING FUNCTION ---
@st.cache_resource(show_spinner=False)
def get_model_and_pipeline():
    if not os.path.exists(MODEL_FILE):
        try:
            if not os.path.exists(DATA_FILE):
                st.error(f"🚨 Critical Error: The file '{DATA_FILE}' was not found.")
                st.stop()
                
            housing = pd.read_csv(DATA_FILE)
            housing['income_cat'] = pd.cut(housing["median_income"], 
                                        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                        labels=[1, 2, 3, 4, 5])
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, _ in split.split(housing, housing['income_cat']):
                strat_train_set = housing.loc[train_index]
            
            housing = strat_train_set.drop("median_house_value", axis=1)
            housing_labels = strat_train_set["median_house_value"].copy()
            housing = housing.drop("income_cat", axis=1)

            num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
            cat_attribs = ["ocean_proximity"]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("std_scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("one_hot", OneHotEncoder(handle_unknown="ignore"))
            ])
            full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", cat_pipeline, cat_attribs)
            ])

            housing_prepared = full_pipeline.fit_transform(housing)
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(housing_prepared, housing_labels)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(full_pipeline, PIPELINE_FILE)
            
        except Exception as e:
            st.error(f"Training Failed: {e}")
            st.stop()

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline

model, pipeline = get_model_and_pipeline()

# ---------------------------------------------------------
# 4. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    if lottie_robot:
        st_lottie(lottie_robot, height=150, key="robot_sidebar")
    st.title("⚙️ Control Center")
    st.markdown("Use controls to adjust values.")
    st.markdown("---")
    st.caption("Developed by **Ajay Savare**")

# ---------------------------------------------------------
# 5. DASHBOARD
# ---------------------------------------------------------
col_head1, col_head2 = st.columns([2, 1])
with col_head1:
    st.title("🏠 AI Estate Pro")
    st.markdown("### Next-Gen Real Estate Price Predictor")
    st.markdown("Enter property details below to get an **Instant Valuation**.")

with col_head2:
    if lottie_house:
        st_lottie(lottie_house, height=160, key="house_anim")

st.markdown("---")

# INPUTS (ALL SET TO 0)
st.subheader("🛠️ Property Configuration")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 📍 Geography")
    # Value 0.0 ani Limits kadhun takle ahet (unlimited input allowed)
    longitude = st.number_input("Longitude", value=0.0, format="%.2f")
    latitude = st.number_input("Latitude", value=0.0, format="%.2f")
    ocean_proximity = st.selectbox("🌊 View Type", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

with c2:
    st.markdown("#### 🏠 Structure")
    # Sliders change karun Number Input kele ahet (typing sathi)
    housing_median_age = st.number_input("Building Age (Years)", value=0)
    total_rooms = st.number_input("Total Rooms", value=0)
    total_bedrooms = st.number_input("Total Bedrooms", value=0)

with c3:
    st.markdown("#### 👥 Demographics")
    population = st.number_input("Area Population", value=0)
    households = st.number_input("Households", value=0)
    median_income = st.number_input("Median Income (Scale 0-15)", value=0.0, format="%.4f")

st.markdown("---")

# PREDICTION
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("🚀 ANALYZE & PREDICT PRICE")

if predict_btn:
    # Check if user entered valid values (Zero asel tar warning deu shakto, pan sadhya run hou dya)
    user_data = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms,
        total_bedrooms, population, households, median_income, ocean_proximity
    ]], columns=[
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
    ])

    with st.spinner('AI Processing...'):
        try:
            prepared_data = pipeline.transform(user_data)
            prediction = model.predict(prepared_data)
            final_price = prediction[0]
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("✅ Valuation Successful!")
    r1, r2 = st.columns([1, 1.5])
    with r1:
        st.markdown(f"<h1 style='color: #4CAF50; font-size: 3.5rem;'>${final_price:,.0f}</h1>", unsafe_allow_html=True)
        st.caption("Estimated Market Value")

    with r2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_price,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Price Meter ($)"},
            gauge = {
                'axis': {'range': [0, 600000], 'tickwidth': 1},
                'bar': {'color': "#DD2476"},
                'steps': [
                    {'range': [0, 200000], 'color': "#a2fca2"},
                    {'range': [200000, 400000], 'color': "#fcfc99"},
                    {'range': [400000, 600000], 'color': "#ff9999"}
                ],
            }
        ))
        fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
    st.balloons()