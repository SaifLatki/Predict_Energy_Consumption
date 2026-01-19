import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configure page
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .prediction-box {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        }
        h1 {
            font-size: 2.5rem;
            margin: 0;
        }
        h2 {
            color: #667eea;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .input-label {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and expected feature names
model_data = joblib.load("energy_model.pkl")

# Handle both old format (model object) and new format (dictionary)
if isinstance(model_data, dict):
    model = model_data["model"]
    expected_features = model_data["features"]
else:
    model = model_data
    try:
        expected_features = joblib.load("feature_names.pkl")
    except:
        st.error("⚠️ Model file is in old format. Please retrain the model using main.py")
        st.stop()

# Header Section
st.markdown("""
    <div class="header-container">
        <h1>⚡ Energy Consumption Predictor</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Predict your building's electricity usage with AI-powered insights</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🏢 Building Configuration")
    st.divider()
    
    # Basic Inputs
    st.markdown("**Physical Characteristics**")
    building_area = st.number_input(
        "Building Area (sqft)",
        value=50000,
        min_value=500,
        max_value=500000,
        step=1000,
        help="Total area of the building in square feet"
    )
    
    st.markdown("**Energy Metrics**")
    peak_demand = st.number_input(
        "Peak Electric Demand (kW)",
        value=1000.0,
        min_value=0.0,
        max_value=10000.0,
        step=10.0,
        help="Maximum demand during peak hours"
    )
    
    gas_usage = st.number_input(
        "Natural Gas Usage (therms)",
        value=5000.0,
        min_value=0.0,
        max_value=50000.0,
        step=100.0,
        help="Monthly natural gas consumption"
    )
    
    energy_intensity = st.number_input(
        "Energy Use Intensity (EUI)",
        value=20.0,
        min_value=0.0,
        max_value=500.0,
        step=1.0,
        help="Energy use per square foot per year"
    )
    
    st.divider()
    st.markdown("**Building Classification**")
    
    building_type = st.selectbox(
        "Building Type",
        ["OFFICE", "RETAIL", "WAREHOUSE", "DININGSERVICE", "DORMITORY",
         "MEDICAL", "LIBRARY", "RECREATION", "COURTHOUSE", "ARTCENTER",
         "CEMETERY", "COMMUNITYCENTER", "CONVENTION_CN", "EXHIBIT",
         "GREENHOUSE", "PARKING"],
        help="Select the primary building type"
    )
    
    department = st.selectbox(
        "Department/Division",
        ["FACILITIES", "FINANCE", "HR", "IT", "OPERATIONS", "OTHER"],
        help="Department responsible for the building"
    )
    
    utility = st.selectbox(
        "Electric Utility Provider",
        ["UTILITY_A", "UTILITY_B", "UTILITY_C"],
        help="Primary electricity provider"
    )

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Input Summary")
    
    # Display input metrics in cards
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666;">Building Area</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #667eea;">{:,.0f}</div>
                <div style="font-size: 0.8rem; color: #999;">sqft</div>
            </div>
        """.format(building_area), unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666;">Peak Demand</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #667eea;">{:,.0f}</div>
                <div style="font-size: 0.8rem; color: #999;">kW</div>
            </div>
        """.format(peak_demand), unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666;">Gas Usage</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #667eea;">{:,.0f}</div>
                <div style="font-size: 0.8rem; color: #999;">therms</div>
            </div>
        """.format(gas_usage), unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666;">Energy Intensity</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #667eea;">{:.1f}</div>
                <div style="font-size: 0.8rem; color: #999;">EUI</div>
            </div>
        """.format(energy_intensity), unsafe_allow_html=True)
    
    st.divider()
    
    # Building details
    st.markdown("### Building Type & Classification")
    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.info(f"🏢 **Type**: {building_type}")
    with detail_cols[1]:
        st.info(f"👥 **Department**: {department}")
    with detail_cols[2]:
        st.info(f"⚡ **Utility**: {utility}")

with col2:
    st.markdown("## 💡 Quick Tips")
    st.markdown("""
    **Optimize Energy Usage:**
    - Upgrade HVAC systems
    - Install LED lighting
    - Improve insulation
    - Use smart controls
    
    **Reduce Peak Demand:**
    - Stagger equipment use
    - Monitor usage patterns
    - Implement demand response
    """)

# Prediction Section
st.divider()
st.markdown("## 🎯 Make Prediction")

col_predict, col_empty = st.columns([2, 1])

with col_predict:
    if st.button("🔮 Predict Electricity Usage", use_container_width=True, key="predict_btn"):
        try:
            # Create base dataframe
            input_df = pd.DataFrame({
                'Building Area': [building_area],
                'Peak Electric Demand': [peak_demand],
                'Natural Gas Usage': [gas_usage],
                'Energy Use Intensity': [energy_intensity],
                'Department': [department],
                'Electric Utility': [utility],
                'Building Type': [building_type]
            })
            
            # One-hot encode categorical features
            input_df = pd.get_dummies(
                input_df,
                columns=['Department', 'Electric Utility', 'Building Type'],
                drop_first=True
            )
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[expected_features]
            
            # Make prediction
            log_pred = model.predict(input_df)
            prediction = np.expm1(log_pred)[0]
            
            # Display prediction with styling
            st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">Predicted Annual Electricity Usage</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 2.8rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        {prediction:,.0f} kWh
                    </h1>
                    <p style="margin: 0.5rem 0; font-size: 0.95rem; opacity: 0.95;">
                        For a {building_area:,.0f} sqft building
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.success("✅ Prediction successful!")
            
            insights_col1, insights_col2, insights_col3 = st.columns(3)
            
            with insights_col1:
                monthly_avg = prediction / 12
                st.metric("Monthly Average", f"{monthly_avg:,.0f} kWh")
            
            with insights_col2:
                intensity = prediction / building_area if building_area > 0 else 0
                st.metric("Usage Intensity", f"{intensity:.2f} kWh/sqft")
            
            with insights_col3:
                est_cost = prediction * 0.12  # Assuming $0.12/kWh
                st.metric("Est. Annual Cost", f"${est_cost:,.0f}")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #999; padding: 1rem; font-size: 0.9rem;">
        🔋 Energy Consumption Predictor | Powered by Machine Learning | 2024
    </div>
""", unsafe_allow_html=True)