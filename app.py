import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please train and save the model first.")
        return None

model = load_model()

# Title and description
st.title("Customer Churn Prediction System")
st.markdown("### Predict customer churn probability based on service usage and demographics")
st.markdown("---")

if model is not None:
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Customer Details", "Prediction Results"])
    
    with tab1:
        # Create columns for input organization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )
            
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )
            
            st.subheader("Phone Service")
            
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )
            
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )
        
        with col2:
            st.subheader("Internet Services")
            
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )
            
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )
            
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )
            
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )
            
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )
            
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )
        
        with col3:
            st.subheader("Account Information")
            
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                help="Number of months the customer has been with the company"
            )
            
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )
            
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )
            
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=0.5
            )
            
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(tenure * monthly_charges),
                step=10.0
            )
        
        st.markdown("---")
        
        # Prediction button
        predict_button = st.button("Predict Churn", type="primary", use_container_width=True)
        
        if predict_button:
            # Create input dataframe - features must match training order
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'SeniorCitizen': [senior_citizen],
                'gender': [gender],
                'Partner': [partner],
                'Dependents': [dependents],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method]
            })
            
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Store results in session state
                st.session_state['prediction'] = prediction
                st.session_state['prediction_proba'] = prediction_proba
                st.session_state['has_prediction'] = True
                
                st.success("Prediction complete! Check the 'Prediction Results' tab.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.session_state['has_prediction'] = False
    
    with tab2:
        if 'has_prediction' in st.session_state and st.session_state['has_prediction']:
            prediction = st.session_state['prediction']
            prediction_proba = st.session_state['prediction_proba']
            
            st.subheader("Prediction Results")
            
            # Create three columns for results
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    label="Churn Prediction",
                    value="WILL CHURN" if prediction == 1 else "WILL STAY",
                    delta="High Risk" if prediction == 1 else "Low Risk",
                    delta_color="inverse"
                )
            
            with res_col2:
                st.metric(
                    label="Churn Probability",
                    value=f"{prediction_proba[1]:.1%}"
                )
            
            with res_col3:
                st.metric(
                    label="Retention Probability",
                    value=f"{prediction_proba[0]:.1%}"
                )
            
            # Visual indicator
            st.markdown("### Risk Assessment")
            churn_prob = prediction_proba[1]
            
            if churn_prob < 0.3:
                st.success("**Low churn risk** - Customer is likely to stay")
                recommendation = """
                **Recommended Actions:**
                - Continue providing excellent service
                - Consider loyalty rewards program
                - Maintain regular communication
                """
            elif churn_prob < 0.6:
                st.warning("**Medium churn risk** - Consider retention strategies")
                recommendation = """
                **Recommended Actions:**
                - Reach out to understand satisfaction level
                - Offer personalized service upgrades
                - Provide special retention offers
                - Improve customer engagement
                """
            else:
                st.error("**High churn risk** - Immediate action recommended")
                recommendation = """
                **Recommended Actions:**
                - Immediate outreach from retention team
                - Offer significant incentives to stay
                - Address service issues promptly
                - Consider contract upgrades with benefits
                - Personalized retention campaign
                """
            
            # Progress bar for visualization
            st.progress(churn_prob)
            
            st.markdown(recommendation)
            
        else:
            st.info("Please enter customer details in the 'Customer Details' tab and click 'Predict Churn' to see results here.")

else:
    st.error("Model file not found. Please train your model first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Customer Churn Prediction System | Built with Streamlit & Scikit-learn</p>
    </div>
""", unsafe_allow_html=True)