import streamlit as st
import pandas as pd
import pymysql
import plotly.express as px
import joblib


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():

    return pymysql.connect(
        host="localhost",
        user="root",
        password="Mani@123",
        port=3306,
        database="tourism_db"
    )


@st.cache_data
def load_data():

    conn = get_connection()

    query = """
    SELECT
        TransactionId,
        UserId,
        VisitYear,
        VisitMonth,
        VisitModeId,
        AttractionId,
        Rating,
        Attraction,
        AttractionType,
        AttractionCity,
        AttractionCountry,
        UserCityName,
        UserCountry,
        UserRegion,
        UserContinent,
        VisitMode
    FROM master_data_table
    """

    data = pd.read_sql(query, conn)

    conn.close()

    return data


df = load_data()


# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    rating_model = joblib.load(
        "tourism_rating_model.pkl"
    )

    rating_columns = joblib.load(
        "tourism_rating_columns.pkl"
    )

    visitmode_model = joblib.load(
        "tourism_visitmode_model.pkl"
    )

    visitmode_columns = joblib.load(
        "tourism_visitmode_columns.pkl"
    )

    attraction_profiles = joblib.load(
        "tourism_attraction_profiles.pkl"
    )

    similarity_matrix = joblib.load(
        "tourism_similarity_matrix.pkl"
    )

    recommendation_ranked = joblib.load(
        "tourism_recommendation_ranked.pkl"
    )

    content_encoder = joblib.load(
        "tourism_content_encoder.pkl"
    )

    return (
        rating_model,
        rating_columns,
        visitmode_model,
        visitmode_columns,
        attraction_profiles,
        similarity_matrix,
        recommendation_ranked,
        content_encoder
    )


(
    rating_model,
    rating_columns,
    visitmode_model,
    visitmode_columns,
    attraction_profiles,
    similarity_matrix,
    recommendation_ranked,
    content_encoder
) = load_models()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🌍 Tourism Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Business Dashboard",
        "⭐ Rating Prediction",
        "👥 Visit Mode Prediction",
        "🎯 Recommendation System"
    ]
)


# ==================================================
# BUSINESS DASHBOARD
# ==================================================

if page == "📊 Business Dashboard":

    st.title("🌍 Tourism Experience Analytics")

    st.caption(
        "Executive dashboard for tourist behaviour, "
        "attraction performance and visitor experience analysis"
    )

    st.divider()

    # ---------------- FILTERS ----------------

    st.sidebar.subheader("Dashboard Filters")

    year_min = int(df["VisitYear"].min())
    year_max = int(df["VisitYear"].max())

    year_range = st.sidebar.slider(
        "Visit Year",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

    rating_range = st.sidebar.slider(
        "Rating",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )

    continent_filter = st.sidebar.multiselect(
        "Tourist Continent",
        sorted(
            df["UserContinent"]
            .dropna()
            .unique()
        )
    )

    country_filter = st.sidebar.multiselect(
        "Tourist Country",
        sorted(
            df["UserCountry"]
            .dropna()
            .unique()
        )
    )

    city_filter = st.sidebar.multiselect(
        "Attraction City",
        sorted(
            df["AttractionCity"]
            .dropna()
            .unique()
        )
    )

    type_filter = st.sidebar.multiselect(
        "Attraction Type",
        sorted(
            df["AttractionType"]
            .dropna()
            .unique()
        )
    )

    mode_filter = st.sidebar.multiselect(
        "Visit Mode",
        sorted(
            df["VisitMode"]
            .dropna()
            .unique()
        )
    )

    top_n = st.sidebar.slider(
        "Top Attractions",
        min_value=5,
        max_value=20,
        value=10
    )

    filtered_df = df[
        (
            df["VisitYear"]
            .between(
                year_range[0],
                year_range[1]
            )
        )
        &
        (
            df["Rating"]
            .between(
                rating_range[0],
                rating_range[1]
            )
        )
    ].copy()

    if continent_filter:
        filtered_df = filtered_df[
            filtered_df[
                "UserContinent"
            ].isin(continent_filter)
        ]

    if country_filter:
        filtered_df = filtered_df[
            filtered_df[
                "UserCountry"
            ].isin(country_filter)
        ]

    if city_filter:
        filtered_df = filtered_df[
            filtered_df[
                "AttractionCity"
            ].isin(city_filter)
        ]

    if type_filter:
        filtered_df = filtered_df[
            filtered_df[
                "AttractionType"
            ].isin(type_filter)
        ]

    if mode_filter:
        filtered_df = filtered_df[
            filtered_df[
                "VisitMode"
            ].isin(mode_filter)
        ]

    if filtered_df.empty:

        st.warning(
            "No records are available "
            "for the selected filters."
        )

        st.stop()

    # ---------------- KPI CARDS ----------------

    total_visits = len(filtered_df)

    unique_tourists = (
        filtered_df["UserId"]
        .nunique()
    )

    total_attractions = (
        filtered_df["AttractionId"]
        .nunique()
    )

    avg_rating = (
        filtered_df["Rating"]
        .mean()
    )

    repeat_visitors = (
        filtered_df
        .groupby("UserId")
        .size()
        .gt(1)
        .sum()
    )

    repeat_percentage = (
        repeat_visitors
        / unique_tourists
        * 100
        if unique_tourists > 0
        else 0
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Total Visits",
        f"{total_visits:,}"
    )

    col2.metric(
        "Unique Tourists",
        f"{unique_tourists:,}"
    )

    col3.metric(
        "Attractions",
        total_attractions
    )

    col4.metric(
        "Average Rating",
        f"{avg_rating:.2f} ⭐"
    )

    col5.metric(
        "Repeat Visitors",
        f"{repeat_percentage:.1f}%"
    )

    st.divider()

    # ---------------- TOP ATTRACTIONS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Top Attractions by Visits"
        )

        top_attractions = (
            filtered_df
            .groupby("Attraction")
            .size()
            .reset_index(
                name="Visits"
            )
            .sort_values(
                "Visits",
                ascending=False
            )
            .head(top_n)
        )

        fig = px.bar(
            top_attractions,
            x="Visits",
            y="Attraction",
            orientation="h",
            text="Visits"
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            },
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- VISIT MODE ----------------

    with col2:

        st.subheader(
            "Visit Mode Distribution"
        )

        visit_mode = (
            filtered_df[
                "VisitMode"
            ]
            .value_counts()
            .reset_index()
        )

        visit_mode.columns = [
            "VisitMode",
            "Visits"
        ]

        fig = px.pie(
            visit_mode,
            names="VisitMode",
            values="Visits",
            hole=0.55
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- YEARLY TREND ----------------

    st.subheader(
        "Tourism Visit Trend"
    )

    yearly_visits = (
        filtered_df
        .groupby("VisitYear")
        .agg(
            Visits=(
                "TransactionId",
                "count"
            ),
            Tourists=(
                "UserId",
                "nunique"
            )
        )
        .reset_index()
    )

    trend_df = yearly_visits.melt(
        id_vars="VisitYear",
        value_vars=[
            "Visits",
            "Tourists"
        ],
        var_name="Metric",
        value_name="Count"
    )

    fig = px.line(
        trend_df,
        x="VisitYear",
        y="Count",
        color="Metric",
        markers=True
    )

    fig.update_layout(
        height=430,
        xaxis_title="Year",
        yaxis_title="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- TYPE ANALYSIS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Visits by Attraction Type"
        )

        type_visits = (
            filtered_df
            .groupby("AttractionType")
            .size()
            .reset_index(
                name="Visits"
            )
            .sort_values(
                "Visits",
                ascending=False
            )
        )

        fig = px.treemap(
            type_visits,
            path=[
                "AttractionType"
            ],
            values="Visits"
        )

        fig.update_layout(
            height=470
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Attraction Type Performance"
        )

        type_rating = (
            filtered_df
            .groupby("AttractionType")
            .agg(
                AverageRating=(
                    "Rating",
                    "mean"
                ),
                Visits=(
                    "TransactionId",
                    "count"
                )
            )
            .reset_index()
        )

        fig = px.scatter(
            type_rating,
            x="Visits",
            y="AverageRating",
            size="Visits",
            hover_name="AttractionType"
        )

        fig.update_layout(
            height=470
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- ORIGIN ANALYSIS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Tourists by Continent"
        )

        continent_data = (
            filtered_df
            .groupby(
                "UserContinent"
            )["UserId"]
            .nunique()
            .reset_index(
                name="Tourists"
            )
            .sort_values(
                "Tourists",
                ascending=False
            )
        )

        fig = px.bar(
            continent_data,
            x="UserContinent",
            y="Tourists",
            text="Tourists"
        )

        fig.update_layout(
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Top Tourist Countries"
        )

        country_data = (
            filtered_df
            .groupby(
                "UserCountry"
            )["UserId"]
            .nunique()
            .reset_index(
                name="Tourists"
            )
            .sort_values(
                "Tourists",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            country_data,
            x="Tourists",
            y="UserCountry",
            orientation="h",
            text="Tourists"
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            },
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- CITY ANALYSIS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Visits by Attraction City"
        )

        city_visits = (
            filtered_df
            .groupby(
                "AttractionCity"
            )
            .size()
            .reset_index(
                name="Visits"
            )
        )

        fig = px.pie(
            city_visits,
            names="AttractionCity",
            values="Visits"
        )

        fig.update_traces(
            textinfo="label+percent"
        )

        fig.update_layout(
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Average Rating by City"
        )

        city_rating = (
            filtered_df
            .groupby(
                "AttractionCity"
            )
            .agg(
                AverageRating=(
                    "Rating",
                    "mean"
                )
            )
            .reset_index()
        )

        fig = px.bar(
            city_rating,
            x="AttractionCity",
            y="AverageRating",
            text_auto=".2f"
        )

        fig.update_layout(
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- RATING ANALYSIS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Rating Distribution"
        )

        rating_data = (
            filtered_df[
                "Rating"
            ]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_data.columns = [
            "Rating",
            "Count"
        ]

        fig = px.bar(
            rating_data,
            x="Rating",
            y="Count",
            text="Count"
        )

        fig.update_layout(
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Average Rating by Visit Mode"
        )

        mode_rating = (
            filtered_df
            .groupby(
                "VisitMode"
            )["Rating"]
            .mean()
            .reset_index(
                name="AverageRating"
            )
            .sort_values(
                "AverageRating",
                ascending=False
            )
        )

        fig = px.bar(
            mode_rating,
            x="VisitMode",
            y="AverageRating",
            text_auto=".2f"
        )

        fig.update_layout(
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- HEATMAP ----------------

    st.subheader(
        "Attraction Type by City"
    )

    heatmap_data = pd.crosstab(
        filtered_df[
            "AttractionType"
        ],
        filtered_df[
            "AttractionCity"
        ]
    )

    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto"
    )

    fig.update_layout(
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- MONTHLY PATTERN ----------------

    st.subheader(
        "Monthly Tourism Pattern"
    )

    monthly_data = (
        filtered_df
        .groupby("VisitMonth")
        .size()
        .reset_index(
            name="Visits"
        )
    )

    fig = px.area(
        monthly_data,
        x="VisitMonth",
        y="Visits",
        markers=True
    )

    fig.update_layout(
        height=400,
        xaxis=dict(
            tickmode="linear",
            dtick=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- ATTRACTION PERFORMANCE ----------------

    st.subheader(
        "Attraction Performance"
    )

    attraction_performance = (
        filtered_df
        .groupby(
            [
                "Attraction",
                "AttractionType",
                "AttractionCity"
            ]
        )
        .agg(
            TotalVisits=(
                "TransactionId",
                "count"
            ),
            UniqueVisitors=(
                "UserId",
                "nunique"
            ),
            AverageRating=(
                "Rating",
                "mean"
            )
        )
        .reset_index()
    )

    attraction_performance[
        "AverageRating"
    ] = (
        attraction_performance[
            "AverageRating"
        ]
        .round(2)
    )

    st.dataframe(
        attraction_performance
        .sort_values(
            [
                "AverageRating",
                "TotalVisits"
            ],
            ascending=[
                False,
                False
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

    # ---------------- DATA EXPLORER ----------------

    st.divider()

    st.subheader(
        "Detailed Tourism Data"
    )

    search_attraction = (
        st.text_input(
            "Search Attraction"
        )
    )

    display_df = (
        filtered_df.copy()
    )

    if search_attraction:

        display_df = display_df[
            display_df[
                "Attraction"
            ]
            .str.contains(
                search_attraction,
                case=False,
                na=False
            )
        ]

    display_columns = [
        "VisitYear",
        "VisitMonth",
        "VisitMode",
        "Attraction",
        "AttractionType",
        "AttractionCity",
        "UserCountry",
        "UserContinent",
        "Rating"
    ]

    st.dataframe(
        display_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
        height=450
    )

    st.caption(
        f"Displaying "
        f"{len(display_df):,} records"
    )


# ==================================================
# RATING PREDICTION
# ==================================================

elif page == "⭐ Rating Prediction":

    st.title(
        "⭐ Attraction Rating Prediction"
    )

    st.caption(
        "Predict the expected tourist rating "
        "for an attraction."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        visit_mode = st.selectbox(
            "Visit Mode",
            sorted(
                df[
                    "VisitMode"
                ]
                .dropna()
                .unique()
            )
        )

        attraction = st.selectbox(
            "Attraction",
            sorted(
                df[
                    "Attraction"
                ]
                .dropna()
                .unique()
            )
        )

        attraction_type = (
            st.selectbox(
                "Attraction Type",
                sorted(
                    df[
                        "AttractionType"
                    ]
                    .dropna()
                    .unique()
                )
            )
        )

        attraction_city = (
            st.selectbox(
                "Attraction City",
                sorted(
                    df[
                        "AttractionCity"
                    ]
                    .dropna()
                    .unique()
                )
            )
        )

    with col2:

        user_country = (
            st.selectbox(
                "Tourist Country",
                sorted(
                    df[
                        "UserCountry"
                    ]
                    .dropna()
                    .unique()
                )
            )
        )

        visit_year = (
            st.slider(
                "Visit Year",
                int(
                    df[
                        "VisitYear"
                    ].min()
                ),
                int(
                    df[
                        "VisitYear"
                    ].max()
                ),
                int(
                    df[
                        "VisitYear"
                    ].max()
                )
            )
        )

        visit_month = (
            st.slider(
                "Visit Month",
                1,
                12,
                6
            )
        )

    if st.button(
        "Predict Rating",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame({
            "VisitMode": [
                visit_mode
            ],
            "Attraction": [
                attraction
            ],
            "AttractionType": [
                attraction_type
            ],
            "AttractionCity": [
                attraction_city
            ],
            "UserCountry": [
                user_country
            ],
            "VisitYear": [
                visit_year
            ],
            "VisitMonth": [
                visit_month
            ]
        })

        categorical_columns = [
            "VisitMode",
            "Attraction",
            "AttractionType",
            "AttractionCity",
            "UserCountry"
        ]

        input_encoded = (
            pd.get_dummies(
                input_df,
                columns=
                categorical_columns,
                dtype=int
            )
        )

        input_encoded = (
            input_encoded
            .reindex(
                columns=
                rating_columns,
                fill_value=0
            )
        )

        predicted_rating = (
            rating_model
            .predict(
                input_encoded
            )[0]
        )

        predicted_rating = max(
            1,
            min(
                5,
                predicted_rating
            )
        )

        st.success(
            f"Predicted Rating: "
            f"⭐ {predicted_rating:.2f} / 5"
        )

        col1, col2 = (
            st.columns(2)
        )

        col1.metric(
            "Expected Rating",
            f"{predicted_rating:.2f}"
        )

        col2.metric(
            "Rating Scale",
            "1 - 5"
        )

        if predicted_rating >= 4.5:

            st.info(
                "Excellent expected "
                "visitor experience."
            )

        elif predicted_rating >= 4:

            st.info(
                "Very good expected "
                "visitor experience."
            )

        elif predicted_rating >= 3:

            st.info(
                "Moderate expected "
                "visitor experience."
            )

        else:

            st.warning(
                "Lower expected "
                "visitor satisfaction."
            )


# ==================================================
# VISIT MODE CLASSIFICATION
# ==================================================

elif page == "👥 Visit Mode Prediction":

    st.title(
        "👥 Tourist Visit Mode Prediction"
    )

    st.caption(
        "Predict whether the visitor "
        "is most likely travelling as "
        "Business, Couples, Family, "
        "Friends or Solo."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        attraction = st.selectbox(
            "Attraction",
            sorted(
                df[
                    "Attraction"
                ]
                .dropna()
                .unique()
            ),
            key="cls_attraction"
        )

        attraction_type = (
            st.selectbox(
                "Attraction Type",
                sorted(
                    df[
                        "AttractionType"
                    ]
                    .dropna()
                    .unique()
                ),
                key="cls_type"
            )
        )

        attraction_city = (
            st.selectbox(
                "Attraction City",
                sorted(
                    df[
                        "AttractionCity"
                    ]
                    .dropna()
                    .unique()
                ),
                key="cls_city"
            )
        )

        visit_year = (
            st.slider(
                "Visit Year",
                int(
                    df[
                        "VisitYear"
                    ].min()
                ),
                int(
                    df[
                        "VisitYear"
                    ].max()
                ),
                int(
                    df[
                        "VisitYear"
                    ].max()
                ),
                key="cls_year"
            )
        )

    with col2:

        user_country = (
            st.selectbox(
                "Tourist Country",
                sorted(
                    df[
                        "UserCountry"
                    ]
                    .dropna()
                    .unique()
                ),
                key="cls_country"
            )
        )

        user_region = (
            st.selectbox(
                "Tourist Region",
                sorted(
                    df[
                        "UserRegion"
                    ]
                    .dropna()
                    .unique()
                ),
                key="cls_region"
            )
        )

        user_continent = (
            st.selectbox(
                "Tourist Continent",
                sorted(
                    df[
                        "UserContinent"
                    ]
                    .dropna()
                    .unique()
                ),
                key="cls_continent"
            )
        )

        visit_month = (
            st.slider(
                "Visit Month",
                1,
                12,
                6,
                key="cls_month"
            )
        )

    if st.button(
        "Predict Visit Mode",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame({
            "VisitYear": [
                visit_year
            ],
            "VisitMonth": [
                visit_month
            ],
            "Attraction": [
                attraction
            ],
            "AttractionType": [
                attraction_type
            ],
            "AttractionCity": [
                attraction_city
            ],
            "UserCountry": [
                user_country
            ],
            "UserRegion": [
                user_region
            ],
            "UserContinent": [
                user_continent
            ]
        })

        categorical_columns = [
            "Attraction",
            "AttractionType",
            "AttractionCity",
            "UserCountry",
            "UserRegion",
            "UserContinent"
        ]

        input_encoded = (
            pd.get_dummies(
                input_df,
                columns=
                categorical_columns,
                dtype=int
            )
        )

        input_encoded = (
            input_encoded
            .reindex(
                columns=
                visitmode_columns,
                fill_value=0
            )
        )

        predicted_mode = (
            visitmode_model
            .predict(
                input_encoded
            )[0]
        )

        probabilities = (
            visitmode_model
            .predict_proba(
                input_encoded
            )[0]
        )

        st.success(
            f"Predicted Visit Mode: "
            f"{predicted_mode}"
        )

        probability_df = (
            pd.DataFrame({
                "Visit Mode":
                visitmode_model.classes_,
                "Probability":
                probabilities
            })
        )

        probability_df = (
            probability_df
            .sort_values(
                "Probability",
                ascending=False
            )
        )

        top_probability = (
            probability_df[
                "Probability"
            ].max()
        )

        col1, col2 = (
            st.columns(2)
        )

        col1.metric(
            "Predicted Mode",
            predicted_mode
        )

        col2.metric(
            "Model Confidence",
            f"{top_probability:.1%}"
        )

        st.subheader(
            "Prediction Probabilities"
        )

        fig = px.bar(
            probability_df,
            x="Probability",
            y="Visit Mode",
            orientation="h",
            text_auto=".1%"
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            },
            xaxis_tickformat=".0%",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Business Suggestion"
        )

        if predicted_mode == "Family":

            st.info(
                "Recommend family-friendly "
                "tour packages, safe activities "
                "and group attractions."
            )

        elif predicted_mode == "Couples":

            st.info(
                "Recommend romantic destinations, "
                "beaches, scenic attractions "
                "and couple packages."
            )

        elif predicted_mode == "Friends":

            st.info(
                "Recommend adventure activities, "
                "group experiences and "
                "entertainment packages."
            )

        elif predicted_mode == "Solo":

            st.info(
                "Recommend flexible itineraries, "
                "cultural experiences and "
                "solo-friendly attractions."
            )

        elif predicted_mode == "Business":

            st.info(
                "Recommend short-duration packages, "
                "premium services and convenient "
                "business travel options."
            )


# ==================================================
# RECOMMENDATION SYSTEM
# ==================================================

elif page == "🎯 Recommendation System":

    st.title(
        "🎯 Smart Attraction Recommendation"
    )

    st.caption(
        "Hybrid recommendation system "
        "combining content similarity, "
        "ratings and attraction popularity."
    )

    st.divider()

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        selected_attraction = (
            st.selectbox(
                "Choose an Attraction You Like",
                sorted(
                    attraction_profiles[
                        "Attraction"
                    ].unique()
                )
            )
        )

    with col2:

        top_n_recommendations = (
            st.slider(
                "Number of Recommendations",
                min_value=3,
                max_value=10,
                value=5
            )
        )

    if st.button(
        "Get Recommendations",
        type="primary",
        use_container_width=True
    ):

        idx = (
            attraction_profiles[
                attraction_profiles[
                    "Attraction"
                ]
                == selected_attraction
            ]
            .index[0]
        )

        attraction_hybrid = (
            attraction_profiles
            .merge(
                recommendation_ranked[
                    [
                        "AttractionId",
                        "VisitCount",
                        "AverageRating",
                        "NormalizedRecommendationScore"
                    ]
                ],
                on="AttractionId",
                how="left"
            )
        )

        attraction_hybrid[
            "SimilarityScore"
        ] = similarity_matrix[idx]

        attraction_hybrid = (
            attraction_hybrid[
                attraction_hybrid[
                    "Attraction"
                ]
                != selected_attraction
            ]
            .copy()
        )

        attraction_hybrid[
            "HybridScore"
        ] = (
            0.70
            * attraction_hybrid[
                "SimilarityScore"
            ]
            +
            0.30
            * attraction_hybrid[
                "NormalizedRecommendationScore"
            ]
        )

        recommendations = (
            attraction_hybrid
            .sort_values(
                "HybridScore",
                ascending=False
            )
            .head(
                top_n_recommendations
            )
        )

        st.subheader(
            f"Recommended Attractions "
            f"Similar to {selected_attraction}"
        )

        for rank, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            with st.container():

                col1, col2, col3, col4 = (
                    st.columns(
                        [3, 2, 1, 1]
                    )
                )

                col1.write(
                    f"**{rank}. "
                    f"{row['Attraction']}**"
                )

                col2.write(
                    f"{row['AttractionType']}  \n"
                    f"{row['AttractionCity']}"
                )

                col3.metric(
                    "Rating",
                    f"{row['AverageRating']:.2f} ⭐"
                )

                col4.metric(
                    "Score",
                    f"{row['HybridScore']:.3f}"
                )

        st.divider()

        chart_df = (
            recommendations[
                [
                    "Attraction",
                    "HybridScore"
                ]
            ]
            .sort_values(
                "HybridScore"
            )
        )

        fig = px.bar(
            chart_df,
            x="HybridScore",
            y="Attraction",
            orientation="h",
            text_auto=".3f"
        )

        fig.update_layout(
            height=430,
            xaxis_title=
            "Recommendation Score",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Recommendation Details"
        )

        recommendation_table = (
            recommendations[
                [
                    "Attraction",
                    "AttractionType",
                    "AttractionCity",
                    "VisitCount",
                    "AverageRating",
                    "SimilarityScore",
                    "HybridScore"
                ]
            ]
            .copy()
        )

        recommendation_table[
            "SimilarityScore"
        ] = (
            recommendation_table[
                "SimilarityScore"
            ]
            .round(3)
        )

        recommendation_table[
            "HybridScore"
        ] = (
            recommendation_table[
                "HybridScore"
            ]
            .round(3)
        )

        st.dataframe(
            recommendation_table,
            use_container_width=True,
            hide_index=True
        )