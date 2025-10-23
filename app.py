import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load data for analysis
@st.cache_data
def load_data():
    try:
        
        df = pd.read_csv('Cleaned data.xls')
        # Clean up the data - remove unnamed index column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
       
        df = df[df['fuel_type'].isin(['Petrol', 'Diesel'])]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to load original model or train new one
def train_model():
    # Try to load the original pickle file directly
    try:
        import warnings
        warnings.filterwarnings('ignore')
        with open('LinearRegressionModel.pkl', 'rb') as f:
            original_model = pickle.load(f)
        st.success("✅ Using original trained model from LinearRegressionModel.pkl")
        return original_model, "original", "original", "original", "original"
    except Exception as e:
        st.warning(f"⚠️ Could not load original model: {str(e)}")
        st.info("🔄 Training a new model from your data...")
    
    # If original model fails, train new one
    df = load_data()
    if df is None:
        return None, None, None, None, None
    
    # Prepare features in the correct order: (name, company, year, kms_driven, fuel_type)
    features = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
    X = df[features].copy()
    y = df['Price']
    
    # Encode categorical variables
    le_name = LabelEncoder()
    le_company = LabelEncoder()
    le_fuel = LabelEncoder()
    
    X['name_encoded'] = le_name.fit_transform(X['name'])
    X['company_encoded'] = le_company.fit_transform(X['company'])
    X['fuel_encoded'] = le_fuel.fit_transform(X['fuel_type'])
    
    # Select final features in correct order: (name, company, year, kms_driven, fuel_type)
    X_final = X[['name_encoded', 'company_encoded', 'year', 'kms_driven', 'fuel_encoded']]
    
    # Train model
    model = LinearRegression()
    model.fit(X_final, y)
    
    # Print model performance for debugging
    print(f"Model R² Score: {model.score(X_final, y):.4f}")
    print(f"Model Intercept: {model.intercept_:.2f}")
    print(f"Model Coefficients: {model.coef_}")
    
    return model, le_name, le_company, le_fuel, features

# Load or train model
model, le_name, le_company, le_fuel, features = train_model()

if model is None:
    st.error("Unable to load or train the model. Please check your data file.")
    st.stop()

# Main header
st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for user inputs
st.sidebar.markdown("### 🎛️ Car Specifications")

# Get unique values from data
df = load_data()
if df is not None:
    companies = sorted(df['company'].unique())
    fuel_types = sorted(df['fuel_type'].unique())
    years = sorted(df['year'].unique())
else:
    companies = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Mahindra', 'Tata', 'BMW', 'Audi', 'Mercedes']
    fuel_types = ['Petrol', 'Diesel']
    years = list(range(2000, 2025))

# User input fields
company = st.sidebar.selectbox("🏭 Company", companies)

# Filter model names based on selected company
df = load_data()
if df is not None:
    available_models = sorted(df[df['company'] == company]['name'].unique()) if company in df['company'].values else []
    if not available_models:
        available_models = ["Select a company first"]
else:
    available_models = ["No data available"]

model_name = st.sidebar.selectbox("🚗 Model Name", available_models)
year = st.sidebar.slider("📅 Year of Purchase", min_value=2000, max_value=2024, value=2015)
kms_driven = st.sidebar.number_input("🛣️ Kilometers Travelled", min_value=0, max_value=500000, value=50000, step=1000)
fuel_type = st.sidebar.selectbox("⛽ Fuel Type", fuel_types)

# Predict button
predict_button = st.sidebar.button("🔮 Predict Price", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="sub-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    
    if predict_button:
        try:
            # Check if valid model name is selected
            if model_name in ["Select a company first", "No data available"]:
                st.warning("Please select a valid model name from the dropdown.")
            else:
                # Check if using original model or new model
                if le_name == "original":
                    # Using original pickle model (Pipeline)
                    st.info("🔄 Using original pipeline model from pickle file...")
                    
                    # Create input data in the format expected by the original pipeline
                    # The pipeline expects raw features, not encoded ones
                    input_data = pd.DataFrame({
                        'name': [model_name],
                        'company': [company],
                        'year': [year],
                        'kms_driven': [kms_driven],
                        'fuel_type': [fuel_type]
                    })
                    
                    # Make prediction using the original pipeline
                    predicted_price = model.predict(input_data)[0]
                else:
                    # Using newly trained model
                    # Prepare input data in correct order: (name, company, year, kms_driven, fuel_type)
                    name_encoded = le_name.transform([model_name])[0]
                    company_encoded = le_company.transform([company])[0]
                    fuel_encoded = le_fuel.transform([fuel_type])[0]
                    
                    # Create feature array in correct order: (name, company, year, kms_driven, fuel_type)
                    input_data = np.array([[name_encoded, company_encoded, year, kms_driven, fuel_encoded]])
                    
                    # Make prediction
                    predicted_price = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 Predicted Price</h2>
                    <h1>₹{predicted_price:,.0f}</h1>
                    <p>Based on your car specifications</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📈 Price Insights")
                
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                
                with col_insight1:
                    st.metric("Model Year", year, f"{2024 - year} years old")
                
                with col_insight2:
                    st.metric("Mileage", f"{kms_driven:,} km", f"{kms_driven/1000:.1f}k km")
                
                with col_insight3:
                    st.metric("Fuel Type", fuel_type, "Efficiency factor")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with col2:
    st.markdown('<div class="sub-header">📋 Car Details</div>', unsafe_allow_html=True)
    
    # Display selected specifications
    st.markdown(f"""
    <div class="metric-container">
        <strong>🏭 Company:</strong> {company}<br>
        <strong>🚗 Model:</strong> {model_name}<br>
        <strong>📅 Year:</strong> {year}<br>
        <strong>🛣️ Kilometers:</strong> {kms_driven:,} km<br>
        <strong>⛽ Fuel Type:</strong> {fuel_type}
    </div>
    """, unsafe_allow_html=True)
    
    # Data insights
    if df is not None:
        st.markdown("### 📊 Data Insights")
        
        # Company distribution
        company_counts = df['company'].value_counts().head(5)
        st.markdown("**Top 5 Companies in Dataset:**")
        for company_name, count in company_counts.items():
            st.markdown(f"• {company_name}: {count} cars")
        
        # Average prices by fuel type
        avg_prices = df.groupby('fuel_type')['Price'].mean()
        st.markdown("**Average Prices by Fuel Type:**")
        for fuel, avg_price in avg_prices.items():
            st.markdown(f"• {fuel}: ₹{avg_price:,.0f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚗 Car Price Predictor | Built with Streamlit & Machine Learning</p>
    <p>Predict car prices based on company, year, mileage, and fuel type</p>
</div>
""", unsafe_allow_html=True)
