import streamlit as st
import pandas as pd
import numpy as np
import pymysql
import joblib
import plotly.express as px
import os

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Tourism Analytics System",
    page_icon="🌍",
    layout="wide"
)

# ===============================
# SAFE PATH LOADING
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rating_model = joblib.load(os.path.join(BASE_DIR, "tourism_rating_model.pkl"))
visitmode_model = joblib.load(os.path.join(BASE_DIR, "tourism_visitmode_classifier.pkl"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))

# ===============================
# MYSQL CONNECTION
# ===============================
def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="Mani@123",
        database="TOURISM",
        port=3306
    )

@st.cache_data
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM MASTER_DATA", conn)
    conn.close()
    return df

master_df = load_data()

# ===============================
# SAFE ENCODING (FIX FOR ERROR)
# ===============================
def safe_encode(col, value):
    le = label_encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return -1   # prevents crash

# ===============================
# SEASON FUNCTION
# ===============================
def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

# ===============================
# SIDEBAR
# ===============================
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "EDA", "Rating Prediction", "Visit Mode Prediction", "Recommendations"]
)

# ===============================
# HOME
# ===============================
if page == "Home":

    st.title("🌍 Tourism Analytics System")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Users", master_df["UserId"].nunique())
    col2.metric("Attractions", master_df["AttractionId"].nunique())
    col3.metric("Countries", master_df["Country"].nunique())
    col4.metric("Avg Rating", round(master_df["Rating"].mean(), 2))

    st.dataframe(master_df.head())

# ===============================
# EDA
# ===============================
#elif page == "EDA":

 #   st.title("📊 EDA Dashboard")
#
 #   st.plotly_chart(px.histogram(master_df, x="Rating"))
  #  st.plotly_chart(px.pie(master_df, names="VisitMode"))
#
 #   top = master_df["AttractionType"].value_counts().head(10).reset_index()
  #  top.columns = ["AttractionType", "Count"]

   # st.plotly_chart(px.bar(top, x="AttractionType", y="Count"))

elif page == "EDA":

    st.title("📊 Tourism EDA Dashboard")

    # ==========================
    # KPI METRICS
    # ==========================
    st.subheader("📌 Key Insights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Users", master_df["UserId"].nunique())
    col2.metric("Attractions", master_df["AttractionId"].nunique())
    col3.metric("Countries", master_df["Country"].nunique())
    col4.metric("Avg Rating", round(master_df["Rating"].mean(), 2))

    st.markdown("---")

    # ==========================
    # 1. RATING DISTRIBUTION
    # ==========================
    st.subheader("⭐ Rating Distribution")

    fig = px.histogram(master_df, x="Rating", nbins=5)
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 2. VISIT MODE DISTRIBUTION
    # ==========================
    st.subheader("✈️ Visit Mode Distribution")

    fig = px.pie(master_df, names="VisitMode")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 3. TOP ATTRACTIONS
    # ==========================
    st.subheader("🏆 Top Attractions")

    top_attractions = master_df["Attraction"].value_counts().head(10)

    fig = px.bar(
        x=top_attractions.index,
        y=top_attractions.values,
        labels={"x": "Attraction", "y": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 4. TOP COUNTRIES
    # ==========================
    st.subheader("🌎 Top Countries by Visits")

    country_visits = master_df["Country"].value_counts().head(10)

    fig = px.bar(
        x=country_visits.index,
        y=country_visits.values,
        labels={"x": "Country", "y": "Visits"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 5. CONTINENT ANALYSIS
    # ==========================
    st.subheader("🌍 Average Rating by Continent")

    cont_rating = master_df.groupby("Continent")["Rating"].mean().sort_values()

    fig = px.bar(
        x=cont_rating.index,
        y=cont_rating.values,
        labels={"x": "Continent", "y": "Average Rating"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 6. MONTHLY TREND
    # ==========================
    st.subheader("📅 Monthly Travel Trend")

    month_trend = master_df["VisitMonth"].value_counts().sort_index()

    fig = px.line(
        x=month_trend.index,
        y=month_trend.values,
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 7. VISIT MODE vs RATING
    # ==========================
    st.subheader("🎯 Visit Mode vs Rating")

    mode_rating = master_df.groupby("VisitMode")["Rating"].mean().sort_values()

    fig = px.bar(
        x=mode_rating.index,
        y=mode_rating.values
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 8. HEATMAP (Country vs Visit Mode)
    # ==========================
    st.subheader("🔥 Country vs Visit Mode Heatmap")

    pivot = pd.pivot_table(
        master_df,
        values="Rating",
        index="Country",
        columns="VisitMode",
        aggfunc="mean"
    ).fillna(0)

    fig = px.imshow(pivot, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 9. USER ACTIVITY DISTRIBUTION
    # ==========================
    st.subheader("👤 User Activity Distribution")

    user_activity = master_df["UserId"].value_counts()

    fig = px.histogram(
        x=user_activity.values,
        nbins=50
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # DATA PREVIEW
    # ==========================
    st.subheader("📄 Sample Data")

    st.dataframe(master_df.head(20))

# ===============================
# RATING PREDICTION
# ===============================
elif page == "Rating Prediction":

    st.title("⭐ Predict Rating")

    user_id = st.number_input("UserId", min_value=1)
    visit_year = st.number_input("Visit Year", 2000, 2035)
    visit_month = st.number_input("Visit Month", 1, 12)

    # SAFE DROPDOWNS (NO ENCODER CRASH)
    continent = st.selectbox("Continent", sorted(master_df["Continent"].dropna().unique()))
    region = st.selectbox("Region", sorted(master_df["Region"].dropna().unique()))
    country = st.selectbox("Country", sorted(master_df["Country"].dropna().unique()))
    city = st.selectbox("City", sorted(master_df["CityName"].dropna().unique()))
    attraction = st.selectbox("Attraction", sorted(master_df["Attraction"].dropna().unique()))
    attraction_type = st.selectbox("Attraction Type", sorted(master_df["AttractionType"].dropna().unique()))
    visit_mode = st.selectbox("Visit Mode", sorted(master_df["VisitMode"].dropna().unique()))

    if st.button("Predict Rating"):

        temp = master_df[master_df["Attraction"] == attraction]

        if temp.empty:
            st.error("Attraction not found")
            st.stop()

        row = temp.iloc[0]

        user_visits = len(master_df[master_df["UserId"] == user_id])
        attraction_popularity = len(master_df[master_df["AttractionId"] == row["AttractionId"]])

        visit_mode_id = master_df[
            master_df["VisitMode"] == visit_mode
        ]["VisitModeId"].iloc[0]

        input_df = pd.DataFrame([{
            "UserId": user_id,
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "VisitModeId": visit_mode_id,
            "AttractionId": row["AttractionId"],
            "ContinentId": row["ContinentId"],
            "RegionId": row["RegionId"],
            "CountryId": row["CountryId"],
            "CityId": row["CityId"],
            "Continent": safe_encode("Continent", continent),
            "Region": safe_encode("Region", region),
            "Country": safe_encode("Country", country),
            "CityName": safe_encode("CityName", city),
            "AttractionCityId": row["AttractionCityId"],
            "AttractionTypeId": row["AttractionTypeId"],
            "Attraction": safe_encode("Attraction", attraction),
            "AttractionType": safe_encode("AttractionType", attraction_type),
            "UserVisitCount": user_visits,
            "AttractionPopularity": attraction_popularity,
            "Season": get_season(visit_month)
        }])

        pred = rating_model.predict(input_df)[0]

        st.success(f"⭐ Predicted Rating: {pred:.2f}")

# ===============================
# VISIT MODE
# ===============================
elif page == "Visit Mode Prediction":

    st.title("✈️ Predict Visit Mode")

    user_id = st.number_input("UserId", min_value=1)
    visit_year = st.number_input("Visit Year", 2000, 2035)
    visit_month = st.number_input("Visit Month", 1, 12)

    attraction = st.selectbox("Attraction", master_df["Attraction"].dropna().unique())

    if st.button("Predict Mode"):

        temp = master_df[master_df["Attraction"] == attraction]

        if temp.empty:
            st.error("No attraction found")
            st.stop()

        row = temp.iloc[0]

        user_visits = len(master_df[master_df["UserId"] == user_id])
        attraction_popularity = len(master_df[master_df["AttractionId"] == row["AttractionId"]])

        input_df = pd.DataFrame([{
            "UserId": user_id,
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "AttractionId": row["AttractionId"],
            "Rating": master_df["Rating"].mean(),
            "ContinentId": row["ContinentId"],
            "RegionId": row["RegionId"],
            "CountryId": row["CountryId"],
            "CityId": row["CityId"],
            "Continent": safe_encode("Continent", row["Continent"]),
            "Region": safe_encode("Region", row["Region"]),
            "Country": safe_encode("Country", row["Country"]),
            "CityName": safe_encode("CityName", row["CityName"]),
            "AttractionCityId": row["AttractionCityId"],
            "AttractionTypeId": row["AttractionTypeId"],
            "Attraction": safe_encode("Attraction", attraction),
            "AttractionType": safe_encode("AttractionType", row["AttractionType"]),
            "AttractionTypeAvgRating": master_df["Rating"].mean(),
            "UserAvgRating": master_df["Rating"].mean(),
            "UserVisitCount": user_visits,
            "AttractionAvgRating": master_df["Rating"].mean(),
            "AttractionPopularity": attraction_popularity,
            "Season": get_season(visit_month)
        }])

        pred = visitmode_model.predict(input_df)[0]

        mode_map = {0:"-",1:"Business",2:"Couples",3:"Family",4:"Friends",5:"Solo"}

        st.success(f"✈️ Predicted Mode: {mode_map.get(pred, pred)}")

# ===============================
# RECOMMENDATION
# ===============================
elif page == "Recommendations":

    st.title("🎯 Content-Based Recommendations")

    user_id = st.selectbox(
        "Select UserId",
        sorted(master_df["UserId"].unique())
    )

    if st.button("Recommend"):

        user_data = master_df[master_df["UserId"] == user_id]

        if user_data.empty:
            st.warning("No history found for this user")
        else:

            # Get user's preferred attraction types
            pref_types = user_data["AttractionType"].value_counts().index.tolist()

            # Score attractions based on similarity
            recs = master_df.copy()

            recs["score"] = recs["AttractionType"].apply(
                lambda x: 1 if x in pref_types else 0
            )

            recs = (
                recs.groupby(["AttractionId", "Attraction"])
                .agg({"score": "sum"})
                .reset_index()
                .sort_values("score", ascending=False)
                .head(10)
            )

            st.dataframe(recs)