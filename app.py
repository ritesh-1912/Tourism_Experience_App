"""
Tourism Experience Analytics - Streamlit Application
Main application for tourism predictions and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

# Resolve project and data directories
BASE_DIR = Path(__file__).parent
_DEFAULT_DATA_DIR = BASE_DIR / 'data'
DATA_DIR = _DEFAULT_DATA_DIR if _DEFAULT_DATA_DIR.exists() else BASE_DIR
MODELS_DIR = BASE_DIR / 'models'  # expected models directory

# Page configuration
st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load processed data"""
    candidate = DATA_DIR / 'processed_data.csv'
    if not candidate.exists():
        # Helpful error showing where we looked
        searched = [str(DATA_DIR / 'processed_data.csv'), str(BASE_DIR / 'processed_data.csv')]
        raise FileNotFoundError(f"Processed data not found. Searched paths: {searched}")
    df = pd.read_csv(candidate)
    return df

@st.cache_resource
def load_models():
    """Load trained models"""
    rm = MODELS_DIR / 'rating_predictor.pkl'
    vm = MODELS_DIR / 'visitmode_classifier.pkl'
    rec = MODELS_DIR / 'recommender.pkl'
    missing = [str(p) for p in (rm, vm, rec) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Model files not found: {missing}")
    rating_model = joblib.load(rm)
    visitmode_model = joblib.load(vm)
    recommender_model = joblib.load(rec)
    return rating_model, visitmode_model, recommender_model

# Load everything
try:
    df = load_data()
    rating_model, visitmode_model, recommender_model = load_models()
    
    # Load lookup tables
    continent_df = pd.read_excel(DATA_DIR / 'Continent.xlsx')
    continent_df = continent_df[continent_df['ContinentId'] != 0]
    
    country_df = pd.read_excel(DATA_DIR / 'Country.xlsx')
    country_df = country_df[country_df['CountryId'] != 0]
    
    region_df = pd.read_excel(DATA_DIR / 'Region.xlsx')
    region_df = region_df[region_df['RegionId'] != 0]
    
    city_df = pd.read_excel(DATA_DIR / 'City.xlsx')
    city_df = city_df[city_df['CityId'] != 0]
    
    attraction_df = pd.read_excel(DATA_DIR / 'Item.xlsx')
    attraction_type_df = pd.read_excel(DATA_DIR / 'Type.xlsx')
    visit_mode_df = pd.read_excel(DATA_DIR / 'Mode.xlsx')
    visit_mode_df = visit_mode_df[visit_mode_df['VisitModeId'] != 0]
    
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# Main header
st.markdown('<div class="main-header">✈️ Tourism Experience Analytics</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">Predict, Classify, and Recommend Tourist Attractions</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🔮 Rating Prediction", "🎯 Visit Mode Classification", 
     "🌟 Recommendations", "📈 Insights"]
)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.markdown('<div class="sub-header">📊 Tourism Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Visits", f"{len(df):,}")
    with col2:
        st.metric("Unique Users", f"{df['UserId'].nunique():,}")
    with col3:
        st.metric("Unique Attractions", f"{df['AttractionId'].nunique():,}")
    with col4:
        st.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    
    st.markdown("---")
    
    # Two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = df['Rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Visit Mode Distribution")
        mode_data = df.merge(visit_mode_df, left_on='VisitMode', right_on='VisitModeId', how='left')
        mode_counts = mode_data['VisitMode_y'].value_counts()
        fig = px.pie(
            values=mode_counts.values,
            names=mode_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.subheader("Visits Over Time")
    temporal_data = df.groupby(['VisitYear', 'VisitMonth']).size().reset_index(name='Count')
    temporal_data['YearMonth'] = temporal_data['VisitYear'].astype(str) + '-' + temporal_data['VisitMonth'].astype(str).str.zfill(2)
    
    fig = px.line(
        temporal_data,
        x='YearMonth',
        y='Count',
        title='Visit Trends',
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top attractions
    st.subheader("Top 10 Most Visited Attractions")
    # Prefer the 'Attraction' column already present in processed data; fallback to merged lookup if needed
    if 'Attraction' in df.columns:
        top_attractions = df['Attraction'].value_counts().head(10)
    else:
        attraction_visits = df.merge(attraction_df, on='AttractionId', how='left')
        if 'Attraction' in attraction_visits.columns:
            top_attractions = attraction_visits['Attraction'].value_counts().head(10)
        elif 'Attraction_x' in attraction_visits.columns:
            top_attractions = attraction_visits['Attraction_x'].value_counts().head(10)
        elif 'Attraction_y' in attraction_visits.columns:
            top_attractions = attraction_visits['Attraction_y'].value_counts().head(10)
        else:
            # Last resort: use AttractionId counts
            top_attractions = df['AttractionId'].value_counts().head(10)

    fig = px.bar(
        x=top_attractions.values,
        y=top_attractions.index.astype(str),
        orientation='h',
        labels={'x': 'Number of Visits', 'y': 'Attraction'},
        color=top_attractions.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: RATING PREDICTION
# ============================================================================
elif page == "🔮 Rating Prediction":
    st.markdown('<div class="sub-header">🔮 Predict Attraction Rating</div>', unsafe_allow_html=True)
    st.write("Enter user and attraction details to predict the rating a user might give.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Details")
        
        # User selection or new user
        user_option = st.radio("Select Option", ["Existing User", "New User"])
        
        if user_option == "Existing User":
            user_id = st.selectbox("Select User ID", df['UserId'].unique())
            user_data = df[df['UserId'] == user_id].iloc[0]
            
            st.info(f"**User Statistics:**\n- Average Rating: {user_data['UserAvgRating']:.2f}\n- Total Visits: {int(user_data['UserTotalVisits'])}\n- Unique Attractions: {int(user_data['UserUniqueAttractions'])}")
        else:
            user_id = df['UserId'].max() + 1
            continent_id = continent_df[continent_df['Continent'] == st.selectbox("Continent", continent_df['Continent'].unique())]['ContinentId'].values[0]
            filtered_regions = region_df[region_df['ContinentId'] == continent_id]
            region_id = filtered_regions[filtered_regions['Region'] == st.selectbox("Region", filtered_regions['Region'].unique())]['RegionId'].values[0]
            filtered_countries = country_df[country_df['RegionId'] == region_id]
            country_id = filtered_countries[filtered_countries['Country'] == st.selectbox("Country", filtered_countries['Country'].unique())]['CountryId'].values[0]
            
            # Use default values for new user
            user_avg_rating = 4.0
            user_total_visits = 1
            user_unique_attractions = 1
    
    with col2:
        st.subheader("Attraction & Visit Details")
        
        attraction_id = st.selectbox("Select Attraction", df['AttractionId'].unique())
        attraction_data = df[df['AttractionId'] == attraction_id].iloc[0]
        attraction_name = attraction_df[attraction_df['AttractionId'] == attraction_id]['Attraction'].values[0]
        
        st.info(f"**{attraction_name}**\n- Average Rating: {attraction_data['AttractionAvgRating']:.2f}\n- Total Visits: {int(attraction_data['AttractionTotalVisits'])}")
        
        visit_year = st.slider("Visit Year", 2020, 2024, 2023)
        visit_month = st.slider("Visit Month", 1, 12, 6)
        visit_mode = st.selectbox("Visit Mode", visit_mode_df['VisitMode'].unique())
        visit_mode_id = visit_mode_df[visit_mode_df['VisitMode'] == visit_mode]['VisitModeId'].values[0]
    
    st.markdown("---")
    
    if st.button("🔮 Predict Rating", use_container_width=True):
        with st.spinner("Predicting rating..."):
            # Prepare input data
            if user_option == "Existing User":
                input_data = pd.DataFrame({
                    'UserId': [user_id],
                    'AttractionId': [attraction_id],
                    'VisitYear': [visit_year],
                    'VisitMonth': [visit_month],
                    'VisitQuarter': [(visit_month-1)//3 + 1],
                    'VisitMode': [visit_mode_id],
                    'ContinentId': [user_data['ContinentId']],
                    'RegionId': [user_data['RegionId']],
                    'CountryId': [user_data['CountryId']],
                    'CityId': [user_data['CityId']],
                    'AttractionCityId': [attraction_data['AttractionCityId']],
                    'AttractionTypeId': [attraction_data['AttractionTypeId']],
                    'UserAvgRating': [user_data['UserAvgRating']],
                    'UserRatingStd': [user_data['UserRatingStd']],
                    'UserTotalVisits': [user_data['UserTotalVisits']],
                    'UserUniqueAttractions': [user_data['UserUniqueAttractions']],
                    'AttractionAvgRating': [attraction_data['AttractionAvgRating']],
                    'AttractionRatingStd': [attraction_data['AttractionRatingStd']],
                    'AttractionTotalVisits': [attraction_data['AttractionTotalVisits']],
                    'AttractionUniqueVisitors': [attraction_data['AttractionUniqueVisitors']],
                    'IsSummerMonth': [1 if visit_month in [6, 7, 8] else 0],
                    'IsWinterMonth': [1 if visit_month in [12, 1, 2] else 0],
                    'ModeTypeCount': [df[(df['VisitMode'] == visit_mode_id) & (df['AttractionTypeId'] == attraction_data['AttractionTypeId'])].shape[0]]
                })
            else:
                # For new user, use defaults
                input_data = pd.DataFrame({
                    'UserId': [user_id],
                    'AttractionId': [attraction_id],
                    'VisitYear': [visit_year],
                    'VisitMonth': [visit_month],
                    'VisitQuarter': [(visit_month-1)//3 + 1],
                    'VisitMode': [visit_mode_id],
                    'ContinentId': [continent_id],
                    'RegionId': [region_id],
                    'CountryId': [country_id],
                    'CityId': [0],
                    'AttractionCityId': [attraction_data['AttractionCityId']],
                    'AttractionTypeId': [attraction_data['AttractionTypeId']],
                    'UserAvgRating': [4.0],
                    'UserRatingStd': [0.5],
                    'UserTotalVisits': [1],
                    'UserUniqueAttractions': [1],
                    'AttractionAvgRating': [attraction_data['AttractionAvgRating']],
                    'AttractionRatingStd': [attraction_data['AttractionRatingStd']],
                    'AttractionTotalVisits': [attraction_data['AttractionTotalVisits']],
                    'AttractionUniqueVisitors': [attraction_data['AttractionUniqueVisitors']],
                    'IsSummerMonth': [1 if visit_month in [6, 7, 8] else 0],
                    'IsWinterMonth': [1 if visit_month in [12, 1, 2] else 0],
                    'ModeTypeCount': [df[(df['VisitMode'] == visit_mode_id) & (df['AttractionTypeId'] == attraction_data['AttractionTypeId'])].shape[0]]
                })
            
            # Scale features
            feature_cols = rating_model['feature_names']
            input_scaled = rating_model['scaler'].transform(input_data[feature_cols])
            
            # Predict
            predicted_rating = rating_model['model'].predict(input_scaled)[0]
            predicted_rating = np.clip(predicted_rating, 1, 5)
            
            # Display result
            st.success("### Prediction Complete!")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{predicted_rating:.2f} / 5.0</h1>", unsafe_allow_html=True)
                
                # Rating interpretation
                if predicted_rating >= 4.5:
                    st.success("🌟 **Excellent!** This user is predicted to love this attraction!")
                elif predicted_rating >= 3.5:
                    st.info("👍 **Good!** This user will likely enjoy this attraction.")
                elif predicted_rating >= 2.5:
                    st.warning("⚠️ **Average.** This attraction might be okay for this user.")
                else:
                    st.error("❌ **Below Average.** This attraction may not be ideal for this user.")

# ============================================================================
# PAGE 3: VISIT MODE CLASSIFICATION
# ============================================================================
elif page == "🎯 Visit Mode Classification":
    st.markdown('<div class="sub-header">🎯 Predict Visit Mode</div>', unsafe_allow_html=True)
    st.write("Predict how a user will travel (Business, Family, Couples, Friends, or Solo).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Details")
        user_id = st.selectbox("Select User ID", df['UserId'].unique(), key='classify_user')
        user_data = df[df['UserId'] == user_id].iloc[0]
        
        st.info(f"**User Statistics:**\n- Average Rating: {user_data['UserAvgRating']:.2f}\n- Total Visits: {int(user_data['UserTotalVisits'])}")
    
    with col2:
        st.subheader("Attraction & Visit Details")
        attraction_id = st.selectbox("Select Attraction", df['AttractionId'].unique(), key='classify_attraction')
        attraction_data = df[df['AttractionId'] == attraction_id].iloc[0]
        attraction_name = attraction_df[attraction_df['AttractionId'] == attraction_id]['Attraction'].values[0]
        
        st.info(f"**{attraction_name}**\n- Type: {attraction_type_df[attraction_type_df['AttractionTypeId'] == attraction_data['AttractionTypeId']]['AttractionType'].values[0]}")
        
        visit_year = st.slider("Visit Year", 2020, 2024, 2023, key='classify_year')
        visit_month = st.slider("Visit Month", 1, 12, 6, key='classify_month')
    
    st.markdown("---")
    
    if st.button("🎯 Predict Visit Mode", use_container_width=True):
        with st.spinner("Predicting visit mode..."):
            # Prepare input data (without VisitMode)
            input_data = pd.DataFrame({
                'UserId': [user_id],
                'AttractionId': [attraction_id],
                'VisitYear': [visit_year],
                'VisitMonth': [visit_month],
                'VisitQuarter': [(visit_month-1)//3 + 1],
                'ContinentId': [user_data['ContinentId']],
                'RegionId': [user_data['RegionId']],
                'CountryId': [user_data['CountryId']],
                'CityId': [user_data['CityId']],
                'AttractionCityId': [attraction_data['AttractionCityId']],
                'AttractionTypeId': [attraction_data['AttractionTypeId']],
                'UserAvgRating': [user_data['UserAvgRating']],
                'UserRatingStd': [user_data['UserRatingStd']],
                'UserTotalVisits': [user_data['UserTotalVisits']],
                'UserUniqueAttractions': [user_data['UserUniqueAttractions']],
                'AttractionAvgRating': [attraction_data['AttractionAvgRating']],
                'AttractionRatingStd': [attraction_data['AttractionRatingStd']],
                'AttractionTotalVisits': [attraction_data['AttractionTotalVisits']],
                'AttractionUniqueVisitors': [attraction_data['AttractionUniqueVisitors']],
                'IsSummerMonth': [1 if visit_month in [6, 7, 8] else 0],
                'IsWinterMonth': [1 if visit_month in [12, 1, 2] else 0],
                'ModeTypeCount': [100]  # Default value
            })
            
            # Scale features
            feature_cols = visitmode_model['feature_names']
            input_scaled = visitmode_model['scaler'].transform(input_data[feature_cols])
            
            # Predict
            predicted_mode_id = visitmode_model['model'].predict(input_scaled)[0]
            predicted_mode = visit_mode_df[visit_mode_df['VisitModeId'] == predicted_mode_id]['VisitMode'].values[0]
            
            # Get prediction probabilities if available
            if hasattr(visitmode_model['model'], 'predict_proba'):
                probabilities = visitmode_model['model'].predict_proba(input_scaled)[0]
                classes = visitmode_model['model'].classes_
                
                # Map to visit mode names
                prob_df = pd.DataFrame({
                    'Visit Mode': [visit_mode_df[visit_mode_df['VisitModeId'] == c]['VisitMode'].values[0] for c in classes],
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
            
            # Display result
            st.success("### Prediction Complete!")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{predicted_mode}</h1>", unsafe_allow_html=True)
                
                # Mode-specific recommendations
                if predicted_mode == 'Business':
                    st.info("💼 **Business Travel** - Consider promoting professional amenities and efficient services.")
                elif predicted_mode == 'Couples':
                    st.info("❤️ **Couples Travel** - Highlight romantic experiences and intimate settings.")
                elif predicted_mode == 'Family':
                    st.info("👨‍👩‍👧‍👦 **Family Travel** - Emphasize family-friendly activities and safety.")
                elif predicted_mode == 'Friends':
                    st.info("👥 **Friends Travel** - Promote group activities and social experiences.")
                elif predicted_mode == 'Solo':
                    st.info("🚶 **Solo Travel** - Focus on self-discovery and flexible options.")
            
            # Show probability distribution if available
            if hasattr(visitmode_model['model'], 'predict_proba'):
                st.markdown("---")
                st.subheader("Prediction Confidence")
                fig = px.bar(
                    prob_df,
                    x='Probability',
                    y='Visit Mode',
                    orientation='h',
                    color='Probability',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: RECOMMENDATIONS
# ============================================================================
elif page == "🌟 Recommendations":
    st.markdown('<div class="sub-header">🌟 Personalized Attraction Recommendations</div>', unsafe_allow_html=True)
    st.write("Get personalized attraction recommendations based on user preferences and history.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("User Selection")
        user_id = st.selectbox("Select User ID", df['UserId'].unique(), key='rec_user')
        
        # Show user's past visits
        user_visits = df[df['UserId'] == user_id].merge(attraction_df, on='AttractionId', how='left')
        st.write(f"**Past Visits:** {len(user_visits)}")
        st.write(f"**Average Rating Given:** {user_visits['Rating'].mean():.2f}")
        
        top_n = st.slider("Number of Recommendations", 3, 10, 5)
        
        rec_type = st.selectbox(
            "Recommendation Method",
            ["Collaborative Filtering", "Content-Based", "Hybrid"]
        )
    
    with col2:
        st.subheader("User's Visit History")
        if len(user_visits) > 0:
            # Build a safe history dataframe using available columns
            history_cols = ['AttractionId', 'Rating', 'Attraction', 'AttractionTypeId']
            history_df = user_visits[[c for c in history_cols if c in user_visits.columns]].head(10)

            # Ensure we have 'Attraction' name
            if 'Attraction' not in history_df.columns and 'AttractionId' in history_df.columns:
                history_df = history_df.merge(attraction_df[['AttractionId', 'Attraction']], on='AttractionId', how='left')

            # Ensure we have 'AttractionTypeId' (may exist in df or in attraction_df)
            if 'AttractionTypeId' not in history_df.columns and 'AttractionId' in history_df.columns:
                history_df = history_df.merge(attraction_df[['AttractionId', 'AttractionTypeId']], on='AttractionId', how='left')

            # Merge to get human-readable AttractionType
            history_df = history_df.merge(attraction_type_df[['AttractionTypeId', 'AttractionType']], on='AttractionTypeId', how='left')

            # Select display columns that exist and rename the type column to 'Type'
            display_cols = [c for c in ['Attraction', 'AttractionType', 'Rating'] if c in history_df.columns]
            display_df = history_df[display_cols].rename(columns={'AttractionType': 'Type'})

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("This user has no visit history.")
    
    st.markdown("---")
    
    if st.button("🌟 Get Recommendations", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            try:
                if rec_type == "Collaborative Filtering":
                    recommendations = recommender_model['user_item_matrix']
                    if user_id in recommendations.index:
                        # Get user's ratings
                        user_ratings = recommendations.loc[user_id]
                        unrated = user_ratings[user_ratings == 0].index.tolist()
                        
                        # Get similar users
                        user_similarity = recommender_model['user_similarity']
                        similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:11]
                        
                        # Calculate scores
                        scores = {}
                        for attraction in unrated:
                            score = 0
                            sim_sum = 0
                            for sim_user, similarity in similar_users.items():
                                if recommendations.loc[sim_user, attraction] > 0:
                                    score += similarity * recommendations.loc[sim_user, attraction]
                                    sim_sum += similarity
                            if sim_sum > 0:
                                scores[attraction] = score / sim_sum
                        
                        # Get top recommendations
                        top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                        
                        if top_recs:
                            rec_df = pd.DataFrame(top_recs, columns=['AttractionId', 'PredictedRating'])
                            rec_df = rec_df.merge(attraction_df, on='AttractionId', how='left')
                            rec_df = rec_df.merge(attraction_type_df, on='AttractionTypeId', how='left')
                            rec_df = rec_df.merge(
                                df.groupby('AttractionId')['Rating'].mean().reset_index().rename(columns={'Rating': 'AvgRating'}),
                                on='AttractionId',
                                how='left'
                            )
                            
                            st.success(f"### Top {len(rec_df)} Recommendations (Collaborative Filtering)")
                            
                            for idx, row in rec_df.iterrows():
                                with st.container():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.markdown(f"**{row['Attraction']}**")
                                        st.caption(f"Type: {row['AttractionType']} | {row['AttractionAddress']}")
                                    with col2:
                                        st.metric("Predicted", f"{row['PredictedRating']:.2f}")
                                    with col3:
                                        st.metric("Avg Rating", f"{row['AvgRating']:.2f}")
                                    st.markdown("---")
                        else:
                            st.warning("No recommendations available for this user.")
                    else:
                        st.warning("User not found in recommendation system.")
                
                elif rec_type == "Content-Based":
                    if user_id in recommender_model['user_item_matrix'].index:
                        # Get user's rated attractions
                        user_ratings = recommender_model['user_item_matrix'].loc[user_id]
                        rated_attractions = user_ratings[user_ratings > 0].index.tolist()
                        
                        if rated_attractions:
                            # Get preferred types
                            preferred_data = df[df['AttractionId'].isin(rated_attractions)]
                            preferred_types = preferred_data['AttractionTypeId'].value_counts().head(3).index.tolist()
                            
                            # Get unrated attractions of preferred types
                            unrated = user_ratings[user_ratings == 0].index.tolist()
                            similar_attractions = df[
                                (df['AttractionId'].isin(unrated)) &
                                (df['AttractionTypeId'].isin(preferred_types))
                            ].groupby('AttractionId').agg({'AttractionAvgRating': 'first'}).reset_index()
                            similar_attractions = similar_attractions.sort_values('AttractionAvgRating', ascending=False).head(top_n)
                            
                            if len(similar_attractions) > 0:
                                rec_df = similar_attractions.merge(attraction_df, on='AttractionId', how='left')
                                rec_df = rec_df.merge(attraction_type_df, on='AttractionTypeId', how='left')
                                
                                st.success(f"### Top {len(rec_df)} Recommendations (Content-Based)")
                                
                                for idx, row in rec_df.iterrows():
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.markdown(f"**{row['Attraction']}**")
                                            st.caption(f"Type: {row['AttractionType']} | {row['AttractionAddress']}")
                                        with col2:
                                            st.metric("Avg Rating", f"{row['AttractionAvgRating']:.2f}")
                                        st.markdown("---")
                            else:
                                st.warning("No content-based recommendations available.")
                        else:
                            st.warning("User has no ratings to base recommendations on.")
                    else:
                        st.warning("User not found in recommendation system.")
                
                else:  # Hybrid
                    st.info("Hybrid recommendations combine both collaborative and content-based approaches.")
                    st.write("Generating combined recommendations...")
                    
                    # Try both methods and combine
                    all_recs = []
                    
                    # Collaborative
                    if user_id in recommender_model['user_item_matrix'].index:
                        recommendations = recommender_model['user_item_matrix']
                        user_ratings = recommendations.loc[user_id]
                        unrated = user_ratings[user_ratings == 0].index.tolist()
                        user_similarity = recommender_model['user_similarity']
                        similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:11]
                        
                        scores = {}
                        for attraction in unrated:
                            score = 0
                            sim_sum = 0
                            for sim_user, similarity in similar_users.items():
                                if recommendations.loc[sim_user, attraction] > 0:
                                    score += similarity * recommendations.loc[sim_user, attraction]
                                    sim_sum += similarity
                            if sim_sum > 0:
                                scores[attraction] = score / sim_sum
                        
                        if scores:
                            all_recs.extend([(aid, score, 'Collaborative') for aid, score in scores.items()])
                    
                    # Content-based
                    if user_id in recommender_model['user_item_matrix'].index:
                        user_ratings = recommender_model['user_item_matrix'].loc[user_id]
                        rated_attractions = user_ratings[user_ratings > 0].index.tolist()
                        
                        if rated_attractions:
                            preferred_data = df[df['AttractionId'].isin(rated_attractions)]
                            preferred_types = preferred_data['AttractionTypeId'].value_counts().head(3).index.tolist()
                            unrated = user_ratings[user_ratings == 0].index.tolist()
                            similar_attractions = df[
                                (df['AttractionId'].isin(unrated)) &
                                (df['AttractionTypeId'].isin(preferred_types))
                            ].groupby('AttractionId').agg({'AttractionAvgRating': 'first'}).reset_index()
                            
                            all_recs.extend([(row['AttractionId'], row['AttractionAvgRating'], 'Content') 
                                           for _, row in similar_attractions.iterrows()])
                    
                    if all_recs:
                        # Combine and deduplicate
                        rec_dict = {}
                        for aid, score, method in all_recs:
                            if aid not in rec_dict or score > rec_dict[aid][0]:
                                rec_dict[aid] = (score, method)
                        
                        # Sort by score
                        top_recs = sorted(rec_dict.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
                        
                        rec_df = pd.DataFrame([
                            {'AttractionId': aid, 'Score': score, 'Method': method}
                            for aid, (score, method) in top_recs
                        ])
                        rec_df = rec_df.merge(attraction_df, on='AttractionId', how='left')
                        rec_df = rec_df.merge(attraction_type_df, on='AttractionTypeId', how='left')
                        
                        st.success(f"### Top {len(rec_df)} Recommendations (Hybrid)")
                        
                        for idx, row in rec_df.iterrows():
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.markdown(f"**{row['Attraction']}**")
                                    st.caption(f"Type: {row['AttractionType']} | {row['AttractionAddress']}")
                                with col2:
                                    st.metric("Score", f"{row['Score']:.2f}")
                                with col3:
                                    st.info(row['Method'])
                                st.markdown("---")
                    else:
                        st.warning("No hybrid recommendations available.")
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# ============================================================================
# PAGE 5: INSIGHTS
# ============================================================================
elif page == "📈 Insights":
    st.markdown('<div class="sub-header">📈 Data Insights & Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Summary", "🌍 Geographic Analysis", "📅 Temporal Patterns", "🎭 Attraction Analysis"])
    
    with tab1:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Statistics**")
            stats = pd.DataFrame({
                'Metric': ['Total Transactions', 'Unique Users', 'Unique Attractions', 'Date Range', 'Avg Rating'],
                'Value': [
                    f"{len(df):,}",
                    f"{df['UserId'].nunique():,}",
                    f"{df['AttractionId'].nunique():,}",
                    f"{df['VisitYear'].min()}-{df['VisitYear'].max()}",
                    f"{df['Rating'].mean():.2f}"
                ]
            })
            st.dataframe(stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Rating Distribution**")
            rating_stats = df['Rating'].describe()
            st.dataframe(rating_stats, use_container_width=True)
        
        st.markdown("---")
        
        # User engagement
        st.subheader("User Engagement Analysis")
        user_stats = df.groupby('UserId').agg({
            'TransactionId': 'count',
            'Rating': 'mean',
            'AttractionId': 'nunique'
        }).rename(columns={
            'TransactionId': 'TotalVisits',
            'Rating': 'AvgRating',
            'AttractionId': 'UniqueAttractions'
        })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.histogram(user_stats, x='TotalVisits', nbins=30, title='Distribution of User Visits')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(user_stats, x='AvgRating', nbins=20, title='Distribution of User Avg Ratings')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(user_stats, x='UniqueAttractions', nbins=30, title='Distribution of Unique Attractions')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Geographic Analysis")
        
        # User distribution by continent
        user_continent = df.merge(continent_df, on='ContinentId', how='left')
        continent_counts = user_continent['Continent'].value_counts()
        
        fig = px.bar(
            x=continent_counts.index,
            y=continent_counts.values,
            title='User Distribution by Continent',
            labels={'x': 'Continent', 'y': 'Number of Visits'},
            color=continent_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top countries
        user_country = df.merge(country_df, on='CountryId', how='left')
        top_countries = user_country['Country'].value_counts().head(15)
        
        fig = px.bar(
            x=top_countries.values,
            y=top_countries.index,
            orientation='h',
            title='Top 15 Countries by Visits',
            labels={'x': 'Number of Visits', 'y': 'Country'},
            color=top_countries.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Temporal Patterns")
        
        # Visits by month
        month_counts = df['VisitMonth'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.bar(
            x=[month_names[i-1] for i in month_counts.index],
            y=month_counts.values,
            title='Visits by Month',
            labels={'x': 'Month', 'y': 'Number of Visits'},
            color=month_counts.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year growth
        year_counts = df['VisitYear'].value_counts().sort_index()
        
        fig = px.line(
            x=year_counts.index,
            y=year_counts.values,
            title='Visits Trend Over Years',
            labels={'x': 'Year', 'y': 'Number of Visits'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Attraction Analysis")
        
        # Attraction type popularity
        attraction_type_data = df.merge(attraction_type_df, on='AttractionTypeId', how='left')
        # Determine which column holds the human-readable attraction type (handle suffixes)
        if 'AttractionType' in attraction_type_data.columns:
            type_col = 'AttractionType'
        elif 'AttractionType_x' in attraction_type_data.columns:
            type_col = 'AttractionType_x'
        elif 'AttractionType_y' in attraction_type_data.columns:
            type_col = 'AttractionType_y'
        else:
            # Ensure we have type names by merging from lookup
            attraction_type_data = attraction_type_data.merge(attraction_type_df[['AttractionTypeId', 'AttractionType']], on='AttractionTypeId', how='left')
            type_col = 'AttractionType'

        type_counts = attraction_type_data[type_col].value_counts()
        
        # Normalize type name and recompute counts for consistent plotting
        attraction_type_data['AttractionTypeName'] = attraction_type_data[type_col]
        type_counts = attraction_type_data['AttractionTypeName'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title='Attraction Type Distribution',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Average rating by attraction type (use normalized name)
            avg_rating_by_type = attraction_type_data.groupby('AttractionTypeName')['Rating'].mean().sort_values(ascending=False)

            fig = px.bar(
                x=avg_rating_by_type.values,
                y=avg_rating_by_type.index,
                orientation='h',
                title='Average Rating by Attraction Type',
                labels={'x': 'Average Rating', 'y': 'Attraction Type'},
                color=avg_rating_by_type.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top attractions
        st.subheader("Top Rated Attractions")
        top_rated = df.groupby('AttractionId').agg({
            'Rating': ['mean', 'count']
        }).reset_index()
        top_rated.columns = ['AttractionId', 'AvgRating', 'NumRatings']
        top_rated = top_rated[top_rated['NumRatings'] >= 10].sort_values('AvgRating', ascending=False).head(10)
        top_rated = top_rated.merge(attraction_df, on='AttractionId', how='left')
        
        st.dataframe(
            top_rated[['Attraction', 'AvgRating', 'NumRatings']].round(2),
            use_container_width=True,
            hide_index=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Tourism Experience Analytics</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
