import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, date
import os
import plotly.graph_objects as go

# Import custom modules
from database import DatabaseManager
from models import AHFPredictionModels
from data_generator import SyntheticDataGenerator
from visualization import RiskVisualizer

# Page configuration
st.set_page_config(
    page_title="AHF Rehospitalization Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize database, models, and other components."""
    db_manager = DatabaseManager()
    models = AHFPredictionModels()
    data_generator = SyntheticDataGenerator()
    visualizer = RiskVisualizer()
    
    # Generate synthetic training data and train models if not already done
    if not models.models_trained():
        with st.spinner("Generating synthetic training data and training models..."):
            training_data = data_generator.generate_training_dataset(n_samples=2000)
            models.train_models(training_data)
    
    return db_manager, models, data_generator, visualizer

def main():
    """Main application function."""
    # Initialize components
    db_manager, models, data_generator, visualizer = initialize_components()
    
    # Title and description
    st.title("🩺 AHF Rehospitalization Risk Predictor")
    st.markdown("""
    **Clinical Decision Support System for 30-Day Acute Heart Failure Readmission Risk**
    
    This system uses NT-proBNP levels, body weight, and ultrasound parameters to predict the risk of 
    30-day rehospitalization in patients with Acute Heart Failure (AHF).
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Risk Assessment", "Batch Upload", "Patient Records", "Patient History", "Model Performance", "Model Explainability", "Data Management"]
    )
    
    if page == "Risk Assessment":
        show_risk_assessment_page(db_manager, models, visualizer)
    elif page == "Batch Upload":
        show_batch_upload_page(db_manager, models, visualizer)
    elif page == "Patient Records":
        show_patient_records_page(db_manager)
    elif page == "Patient History":
        show_patient_history_page(db_manager, visualizer)
    elif page == "Model Performance":
        show_model_performance_page(models, visualizer)
    elif page == "Model Explainability":
        show_model_explainability_page(models, visualizer)
    elif page == "Data Management":
        show_data_management_page(db_manager, data_generator)

def show_risk_assessment_page(db_manager, models, visualizer):
    """Display the risk assessment page."""
    st.header("Patient Risk Assessment")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        patient_id = st.text_input("Patient ID", value=f"PT{np.random.randint(1000, 9999)}")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Body Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        
        st.subheader("Laboratory Results")
        nt_probnp = st.number_input("NT-proBNP (pg/mL)", min_value=0.0, max_value=50000.0, value=1500.0, step=10.0)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=1.2, step=0.1)
        
    with col2:
        st.subheader("Ultrasound Parameters")
        b_line_score = st.number_input("B-line Score", min_value=0, max_value=28, value=8)
        ivc_collapsibility = st.number_input("IVC Collapsibility (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        
        st.subheader("Clinical Parameters")
        ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=10.0, max_value=80.0, value=35.0, step=1.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
        
        # Comorbidities
        diabetes = st.checkbox("Diabetes Mellitus")
        hypertension = st.checkbox("Hypertension")
        ckd = st.checkbox("Chronic Kidney Disease")
        afib = st.checkbox("Atrial Fibrillation")
    
    # Risk assessment button
    if st.button("Calculate Risk", type="primary"):
        # Prepare patient data
        patient_data = {
            'age': age,
            'gender': 1 if gender == 'Male' else 0,
            'weight': weight,
            'nt_probnp': nt_probnp,
            'creatinine': creatinine,
            'b_line_score': b_line_score,
            'ivc_collapsibility': ivc_collapsibility,
            'ejection_fraction': ejection_fraction,
            'systolic_bp': systolic_bp,
            'heart_rate': heart_rate,
            'diabetes': int(diabetes),
            'hypertension': int(hypertension),
            'ckd': int(ckd),
            'afib': int(afib)
        }
        
        # Get predictions
        predictions = models.predict_risk(patient_data)
        
        # Store in session state for saving later
        st.session_state['last_assessment'] = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'weight': weight,
            'nt_probnp': nt_probnp,
            'creatinine': creatinine,
            'b_line_score': b_line_score,
            'ivc_collapsibility': ivc_collapsibility,
            'ejection_fraction': ejection_fraction,
            'systolic_bp': systolic_bp,
            'heart_rate': heart_rate,
            'diabetes': int(diabetes),
            'hypertension': int(hypertension),
            'ckd': int(ckd),
            'afib': int(afib),
            'lr_probability': predictions['logistic_regression']['probability'],
            'xgb_probability': predictions['xgboost']['probability'],
            'ensemble_probability': (predictions['logistic_regression']['probability'] + predictions['xgboost']['probability']) / 2,
            'risk_level': "High Risk" if (predictions['logistic_regression']['probability'] + predictions['xgboost']['probability']) / 2 > 0.5 else "Low Risk",
            'patient_data': patient_data
        }
        
        # Display results
        st.subheader("Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Logistic Regression Results
            lr_prob = predictions['logistic_regression']['probability']
            lr_risk = predictions['logistic_regression']['risk_level']
            st.metric(
                label="Logistic Regression",
                value=f"{lr_prob:.1%}",
                delta=lr_risk
            )
            
        with col2:
            # XGBoost Results
            xgb_prob = predictions['xgboost']['probability']
            xgb_risk = predictions['xgboost']['risk_level']
            st.metric(
                label="XGBoost",
                value=f"{xgb_prob:.1%}",
                delta=xgb_risk
            )
            
        with col3:
            # Ensemble Result
            ensemble_prob = (lr_prob + xgb_prob) / 2
            ensemble_risk = "High Risk" if ensemble_prob > 0.5 else "Low Risk"
            st.metric(
                label="Ensemble Average",
                value=f"{ensemble_prob:.1%}",
                delta=ensemble_risk
            )
        
        # Visualization
        visualizer.plot_risk_gauge(ensemble_prob)
        
        # Risk interpretation
        st.subheader("Clinical Interpretation")
        if ensemble_prob > 0.7:
            st.error("""
            **High Risk (>70%)**: This patient has a high probability of 30-day readmission. 
            Consider intensive monitoring, early follow-up, and optimization of heart failure therapy.
            """)
        elif ensemble_prob > 0.5:
            st.warning("""
            **Moderate Risk (50-70%)**: This patient has moderate risk of readmission. 
            Schedule close follow-up and ensure medication compliance.
            """)
        else:
            st.success("""
            **Low Risk (<50%)**: This patient has low probability of 30-day readmission. 
            Standard follow-up care is appropriate.
            """)
        
        # Feature importance
        st.subheader("Key Risk Factors")
        feature_importance = models.get_feature_importance()
        visualizer.plot_feature_importance(feature_importance, st.session_state['last_assessment']['patient_data'])
    
    # Save button - outside calculate risk block so it persists
    if 'last_assessment' in st.session_state and st.session_state['last_assessment'] is not None:
        if st.button("Save Assessment", key="save_button"):
            assessment = st.session_state['last_assessment']
            record_data = {
                'patient_id': assessment['patient_id'],
                'assessment_date': datetime.now().isoformat(),
                'age': assessment['age'],
                'gender': assessment['gender'],
                'weight': assessment['weight'],
                'nt_probnp': assessment['nt_probnp'],
                'creatinine': assessment['creatinine'],
                'b_line_score': assessment['b_line_score'],
                'ivc_collapsibility': assessment['ivc_collapsibility'],
                'ejection_fraction': assessment['ejection_fraction'],
                'systolic_bp': assessment['systolic_bp'],
                'heart_rate': assessment['heart_rate'],
                'diabetes': assessment['diabetes'],
                'hypertension': assessment['hypertension'],
                'ckd': assessment['ckd'],
                'afib': assessment['afib'],
                'lr_probability': assessment['lr_probability'],
                'xgb_probability': assessment['xgb_probability'],
                'ensemble_probability': assessment['ensemble_probability'],
                'risk_level': assessment['risk_level']
            }
            
            db_manager.save_assessment(record_data)
            st.success("✓ Assessment saved successfully to database!")
            st.session_state['last_assessment'] = None  # Clear after saving

def show_patient_records_page(db_manager):
    """Display patient records page."""
    st.header("Patient Records")
    
    # Get all assessments
    assessments = db_manager.get_all_assessments()
    
    if assessments:
        df = pd.DataFrame(assessments)
        df['assessment_date'] = pd.to_datetime(df['assessment_date']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assessments", len(df))
        with col2:
            high_risk_count = len(df[df['ensemble_probability'] > 0.5])
            st.metric("High Risk Patients", high_risk_count)
        with col3:
            avg_risk = df['ensemble_probability'].mean()
            st.metric("Average Risk", f"{avg_risk:.1%}")
        with col4:
            avg_nt_probnp = df['nt_probnp'].mean()
            st.metric("Avg NT-proBNP", f"{avg_nt_probnp:.0f} pg/mL")
        
        # Display table
        st.subheader("Assessment History")
        display_columns = [
            'patient_id', 'assessment_date', 'age', 'gender', 'nt_probnp',
            'ensemble_probability', 'risk_level'
        ]
        st.dataframe(
            df[display_columns],
            use_container_width=True,
            column_config={
                'ensemble_probability': st.column_config.ProgressColumn(
                    'Risk Probability',
                    help='30-day readmission risk probability',
                    format='%.1%%',
                    min_value=0,
                    max_value=1
                )
            }
        )
        
        # Export functionality
        if st.button("Export Records"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ahf_assessments_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No patient assessments found. Complete a risk assessment to see records here.")

def show_model_performance_page(models, visualizer):
    """Display model performance metrics."""
    st.header("Model Performance Metrics")
    
    if models.models_trained():
        performance = models.get_performance_metrics()
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            lr_metrics = performance['logistic_regression']
            st.metric("Accuracy", f"{lr_metrics['accuracy']:.3f}")
            st.metric("AUC-ROC", f"{lr_metrics['auc']:.3f}")
            st.metric("Sensitivity", f"{lr_metrics['sensitivity']:.3f}")
            st.metric("Specificity", f"{lr_metrics['specificity']:.3f}")
            
        with col2:
            st.subheader("XGBoost")
            xgb_metrics = performance['xgboost']
            st.metric("Accuracy", f"{xgb_metrics['accuracy']:.3f}")
            st.metric("AUC-ROC", f"{xgb_metrics['auc']:.3f}")
            st.metric("Sensitivity", f"{xgb_metrics['sensitivity']:.3f}")
            st.metric("Specificity", f"{xgb_metrics['specificity']:.3f}")
        
        # ROC Curve visualization
        st.subheader("ROC Curves")
        visualizer.plot_roc_curves(performance)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        visualizer.plot_confusion_matrices(performance)
        
    else:
        st.warning("Models not trained yet. Please wait for initialization to complete.")

def show_model_explainability_page(models, visualizer):
    """Display model explainability and feature interpretation."""
    st.header("Model Explainability & Clinical Interpretation")
    
    st.markdown("""
    This page provides detailed insights into how the prediction models make decisions,
    helping clinicians understand which factors contribute most to rehospitalization risk.
    """)
    
    if not models.models_trained():
        st.warning("Models not trained yet. Please wait for initialization to complete.")
        return
    
    # Model comparison
    st.subheader("Model Comparison")
    
    performance = models.get_performance_metrics()
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Accuracy', 'AUC-ROC', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score'],
        'Logistic Regression': [
            f"{performance['logistic_regression']['accuracy']:.3f}",
            f"{performance['logistic_regression']['auc']:.3f}",
            f"{performance['logistic_regression']['sensitivity']:.3f}",
            f"{performance['logistic_regression']['specificity']:.3f}",
            f"{performance['logistic_regression']['precision']:.3f}",
            f"{performance['logistic_regression']['f1']:.3f}"
        ],
        'XGBoost': [
            f"{performance['xgboost']['accuracy']:.3f}",
            f"{performance['xgboost']['auc']:.3f}",
            f"{performance['xgboost']['sensitivity']:.3f}",
            f"{performance['xgboost']['specificity']:.3f}",
            f"{performance['xgboost']['precision']:.3f}",
            f"{performance['xgboost']['f1']:.3f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    
    importance_dict = models.get_feature_importance()
    
    if importance_dict:
        # Create visualization
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # Map to clinical names
        feature_labels = {
            'nt_probnp': 'NT-proBNP',
            'age': 'Age',
            'ejection_fraction': 'Ejection Fraction',
            'creatinine': 'Creatinine',
            'b_line_score': 'B-line Score',
            'ivc_collapsibility': 'IVC Collapsibility',
            'weight': 'Body Weight',
            'systolic_bp': 'Systolic BP',
            'heart_rate': 'Heart Rate',
            'diabetes': 'Diabetes',
            'hypertension': 'Hypertension',
            'ckd': 'Chronic Kidney Disease',
            'afib': 'Atrial Fibrillation',
            'gender': 'Gender'
        }
        
        df_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'Clinical_Name': [feature_labels.get(f, f) for f in features]
        })
        df_importance = df_importance.sort_values('Importance', ascending=True)
        
        # Plot
        fig = go.Figure(go.Bar(
            x=df_importance['Importance'],
            y=df_importance['Clinical_Name'],
            orientation='h',
            marker=dict(
                color=df_importance['Importance'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            )
        ))
        
        fig.update_layout(
            title='Feature Importance (XGBoost Model)',
            xaxis_title='Importance Score',
            yaxis_title='Clinical Features',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical interpretation
        st.subheader("Clinical Interpretation Guide")
        
        # Get top 5 features
        top_features = df_importance.nlargest(5, 'Importance')
        
        st.markdown("**Top 5 Most Important Risk Factors:**")
        
        for idx, row in top_features.iterrows():
            feature = row['Feature']
            clinical_name = row['Clinical_Name']
            importance = row['Importance']
            
            # Clinical interpretations
            interpretations = {
                'nt_probnp': "NT-proBNP is a biomarker for heart failure severity. Higher levels indicate increased cardiac stress and are strongly associated with worse outcomes.",
                'age': "Advanced age is associated with increased cardiovascular risk, reduced physiological reserve, and higher rates of comorbidities.",
                'ejection_fraction': "Lower ejection fraction indicates reduced cardiac pump function, a key predictor of heart failure outcomes.",
                'creatinine': "Elevated creatinine reflects kidney dysfunction, which often accompanies heart failure and predicts worse prognosis.",
                'b_line_score': "B-lines on ultrasound indicate pulmonary congestion/edema, a sign of fluid overload in heart failure.",
                'ivc_collapsibility': "Reduced IVC collapsibility suggests elevated right atrial pressure and volume overload.",
                'ckd': "Chronic kidney disease frequently coexists with heart failure and significantly increases readmission risk.",
                'diabetes': "Diabetes is a major cardiovascular risk factor and complicates heart failure management.",
                'afib': "Atrial fibrillation increases thromboembolic risk and is associated with worse heart failure outcomes.",
                'weight': "Rapid weight gain may indicate fluid retention, a warning sign of decompensating heart failure.",
                'systolic_bp': "Blood pressure abnormalities (very high or very low) can indicate inadequate heart failure control.",
                'heart_rate': "Elevated heart rate may indicate inadequate rate control or worsening heart failure.",
                'hypertension': "Hypertension increases cardiac workload and contributes to heart failure progression.",
                'gender': "Gender differences in heart failure presentation, treatment response, and outcomes are well documented."
            }
            
            interpretation = interpretations.get(feature, "This factor contributes to the overall risk assessment.")
            
            with st.expander(f"{idx + 1}. {clinical_name} (Importance: {importance:.3f})"):
                st.write(interpretation)
        
        # Logistic regression coefficients
        st.subheader("Logistic Regression Coefficients")
        
        st.markdown("""
        The coefficients below show how each feature influences the log-odds of 30-day rehospitalization.
        - **Positive coefficients**: increase risk
        - **Negative coefficients**: decrease risk
        - **Larger absolute values**: stronger effect
        """)
        
        if hasattr(models.logistic_model, 'coef_'):
            coefficients = models.logistic_model.coef_[0]
            coef_df = pd.DataFrame({
                'Feature': [feature_labels.get(f, f) for f in models.feature_names],
                'Coefficient': coefficients,
                'Effect': ['Increases Risk' if c > 0 else 'Decreases Risk' for c in coefficients],
                'Magnitude': np.abs(coefficients)
            })
            coef_df = coef_df.sort_values('Magnitude', ascending=False)
            
            # Plot coefficients
            fig_coef = go.Figure(go.Bar(
                x=coef_df['Coefficient'],
                y=coef_df['Feature'],
                orientation='h',
                marker=dict(
                    color=coef_df['Coefficient'],
                    colorscale='RdBu_r',
                    showscale=True,
                    colorbar=dict(title="Coefficient Value"),
                    cmid=0
                ),
                text=[f"{c:+.3f}" for c in coef_df['Coefficient']],
                textposition='auto'
            ))
            
            fig_coef.update_layout(
                title='Logistic Regression Coefficients',
                xaxis_title='Coefficient Value',
                yaxis_title='Clinical Features',
                height=500
            )
            
            st.plotly_chart(fig_coef, use_container_width=True)
        
        # Model usage recommendations
        st.subheader("Clinical Decision Support Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**When to Use This Model:**")
            st.write("✓ Risk stratification at hospital discharge")
            st.write("✓ Identifying patients needing intensive follow-up")
            st.write("✓ Optimizing resource allocation")
            st.write("✓ Supporting clinical decision-making")
        
        with col2:
            st.markdown("**Model Limitations:**")
            st.write("⚠️ Trained on synthetic data - requires validation with real patient data")
            st.write("⚠️ Does not replace clinical judgment")
            st.write("⚠️ May not capture all relevant clinical factors")
            st.write("⚠️ Population-specific calibration may be needed")
        
        # Risk threshold guidance
        st.subheader("Risk Threshold Interpretation")
        
        threshold_guide = pd.DataFrame({
            'Risk Category': ['Low Risk', 'Moderate Risk', 'High Risk'],
            'Probability Range': ['< 30%', '30% - 70%', '> 70%'],
            'Clinical Action': [
                'Standard follow-up care, routine monitoring',
                'Enhanced follow-up, medication review, lifestyle counseling',
                'Intensive monitoring, early clinic visit, home health services, consider telemonitoring'
            ],
            'Follow-up Timeline': [
                '2-4 weeks',
                '1-2 weeks',
                '3-7 days or earlier'
            ]
        })
        
        st.dataframe(threshold_guide, use_container_width=True)

def show_batch_upload_page(db_manager, models, visualizer):
    """Display batch upload page for CSV processing."""
    st.header("Batch Patient Risk Assessment")
    
    st.markdown("""
    Upload a CSV file containing multiple patient records to perform batch risk assessments.
    
    **Required CSV columns:**
    - patient_id, age, gender (Male/Female), weight, nt_probnp, creatinine
    - b_line_score, ivc_collapsibility, ejection_fraction, systolic_bp, heart_rate
    - diabetes (0/1), hypertension (0/1), ckd (0/1), afib (0/1)
    """)
    
    # Sample CSV download
    st.subheader("Download Sample CSV Template")
    sample_data = {
        'patient_id': ['PT001', 'PT002', 'PT003'],
        'age': [65, 72, 58],
        'gender': ['Male', 'Female', 'Male'],
        'weight': [75.0, 68.5, 82.3],
        'nt_probnp': [1500, 3200, 890],
        'creatinine': [1.2, 1.8, 1.0],
        'b_line_score': [8, 15, 5],
        'ivc_collapsibility': [45.0, 28.0, 55.0],
        'ejection_fraction': [35.0, 28.0, 42.0],
        'systolic_bp': [120, 110, 135],
        'heart_rate': [80, 95, 72],
        'diabetes': [1, 1, 0],
        'hypertension': [1, 1, 1],
        'ckd': [0, 1, 0],
        'afib': [0, 1, 0]
    }
    sample_df = pd.DataFrame(sample_data)
    csv_template = sample_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="patient_batch_template.csv",
        mime="text/csv"
    )
    
    # File upload
    st.subheader("Upload Patient Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ File uploaded successfully! Found {len(batch_df)} patient records.")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(batch_df.head(10))
            
            # Validate columns
            required_columns = [
                'patient_id', 'age', 'gender', 'weight', 'nt_probnp', 'creatinine',
                'b_line_score', 'ivc_collapsibility', 'ejection_fraction',
                'systolic_bp', 'heart_rate', 'diabetes', 'hypertension', 'ckd', 'afib'
            ]
            
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                if st.button("🚀 Process Batch Assessment", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, row in batch_df.iterrows():
                        # Prepare patient data
                        patient_data = {
                            'age': int(row['age']),
                            'gender': 1 if row['gender'].lower() == 'male' else 0,
                            'weight': float(row['weight']),
                            'nt_probnp': float(row['nt_probnp']),
                            'creatinine': float(row['creatinine']),
                            'b_line_score': int(row['b_line_score']),
                            'ivc_collapsibility': float(row['ivc_collapsibility']),
                            'ejection_fraction': float(row['ejection_fraction']),
                            'systolic_bp': int(row['systolic_bp']),
                            'heart_rate': int(row['heart_rate']),
                            'diabetes': int(row['diabetes']),
                            'hypertension': int(row['hypertension']),
                            'ckd': int(row['ckd']),
                            'afib': int(row['afib'])
                        }
                        
                        # Get predictions
                        predictions = models.predict_risk(patient_data)
                        
                        lr_prob = predictions['logistic_regression']['probability']
                        xgb_prob = predictions['xgboost']['probability']
                        ensemble_prob = (lr_prob + xgb_prob) / 2
                        ensemble_risk = "High Risk" if ensemble_prob > 0.5 else "Low Risk"
                        
                        # Save to database
                        record_data = {
                            'patient_id': row['patient_id'],
                            'assessment_date': datetime.now().isoformat(),
                            'age': int(row['age']),
                            'gender': row['gender'],
                            'weight': float(row['weight']),
                            'nt_probnp': float(row['nt_probnp']),
                            'creatinine': float(row['creatinine']),
                            'b_line_score': int(row['b_line_score']),
                            'ivc_collapsibility': float(row['ivc_collapsibility']),
                            'ejection_fraction': float(row['ejection_fraction']),
                            'systolic_bp': int(row['systolic_bp']),
                            'heart_rate': int(row['heart_rate']),
                            'diabetes': int(row['diabetes']),
                            'hypertension': int(row['hypertension']),
                            'ckd': int(row['ckd']),
                            'afib': int(row['afib']),
                            'lr_probability': lr_prob,
                            'xgb_probability': xgb_prob,
                            'ensemble_probability': ensemble_prob,
                            'risk_level': ensemble_risk
                        }
                        
                        db_manager.save_assessment(record_data)
                        
                        # Store result
                        results.append({
                            'patient_id': row['patient_id'],
                            'age': row['age'],
                            'nt_probnp': row['nt_probnp'],
                            'lr_probability': f"{lr_prob:.1%}",
                            'xgb_probability': f"{xgb_prob:.1%}",
                            'ensemble_probability': f"{ensemble_prob:.1%}",
                            'risk_level': ensemble_risk
                        })
                        
                        # Update progress
                        progress = (idx + 1) / len(batch_df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing patient {idx + 1} of {len(batch_df)}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✓ Successfully processed {len(results)} patients!")
                    
                    # Display results
                    st.subheader("Batch Assessment Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    results_csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=results_csv,
                        file_name=f"batch_assessment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        high_risk_count = sum(1 for r in results if r['risk_level'] == 'High Risk')
                        st.metric("High Risk Patients", high_risk_count)
                    with col2:
                        low_risk_count = sum(1 for r in results if r['risk_level'] == 'Low Risk')
                        st.metric("Low Risk Patients", low_risk_count)
                    with col3:
                        avg_risk = sum(float(r['ensemble_probability'].strip('%'))/100 for r in results) / len(results)
                        st.metric("Average Risk", f"{avg_risk:.1%}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_patient_history_page(db_manager, visualizer):
    """Display patient history with longitudinal trends."""
    st.header("Patient History & Trends")
    
    # Get all unique patients
    assessments = db_manager.get_all_assessments()
    
    if not assessments:
        st.info("No patient assessments found. Complete a risk assessment to see patient history.")
        return
    
    df = pd.DataFrame(assessments)
    unique_patients = df['patient_id'].unique()
    
    # Patient selection
    selected_patient = st.selectbox(
        "Select Patient ID",
        options=unique_patients,
        help="Choose a patient to view their assessment history and trends"
    )
    
    if selected_patient:
        # Get patient history
        patient_history = db_manager.get_assessment_by_patient_id(selected_patient)
        patient_df = pd.DataFrame(patient_history)
        patient_df['assessment_date'] = pd.to_datetime(patient_df['assessment_date'])
        patient_df = patient_df.sort_values('assessment_date')
        
        # Patient summary
        st.subheader(f"Patient {selected_patient} - Assessment History")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assessments", len(patient_df))
        with col2:
            latest_risk = patient_df.iloc[-1]['ensemble_probability']
            st.metric("Latest Risk", f"{latest_risk:.1%}")
        with col3:
            avg_nt_probnp = patient_df['nt_probnp'].mean()
            st.metric("Avg NT-proBNP", f"{avg_nt_probnp:.0f} pg/mL")
        with col4:
            latest_weight = patient_df.iloc[-1]['weight']
            st.metric("Latest Weight", f"{latest_weight:.1f} kg")
        
        # Display assessment table
        st.subheader("Assessment Records")
        display_cols = ['assessment_date', 'nt_probnp', 'weight', 'b_line_score', 
                       'ivc_collapsibility', 'ensemble_probability', 'risk_level']
        display_df = patient_df[display_cols].copy()
        display_df['assessment_date'] = display_df['assessment_date'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)
        
        # Longitudinal trend visualizations
        if len(patient_df) > 1:
            st.subheader("Biomarker Trends Over Time")
            
            # NT-proBNP trend
            fig_nt = go.Figure()
            fig_nt.add_trace(go.Scatter(
                x=patient_df['assessment_date'],
                y=patient_df['nt_probnp'],
                mode='lines+markers',
                name='NT-proBNP',
                line=dict(color='#007bff', width=2),
                marker=dict(size=8)
            ))
            fig_nt.update_layout(
                title='NT-proBNP Levels Over Time',
                xaxis_title='Assessment Date',
                yaxis_title='NT-proBNP (pg/mL)',
                height=400
            )
            st.plotly_chart(fig_nt, use_container_width=True)
            
            # Weight trend
            fig_weight = go.Figure()
            fig_weight.add_trace(go.Scatter(
                x=patient_df['assessment_date'],
                y=patient_df['weight'],
                mode='lines+markers',
                name='Body Weight',
                line=dict(color='#28a745', width=2),
                marker=dict(size=8)
            ))
            fig_weight.update_layout(
                title='Body Weight Changes Over Time',
                xaxis_title='Assessment Date',
                yaxis_title='Weight (kg)',
                height=400
            )
            st.plotly_chart(fig_weight, use_container_width=True)
            
            # Risk trajectory
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=patient_df['assessment_date'],
                y=patient_df['ensemble_probability'],
                mode='lines+markers',
                name='Risk Probability',
                line=dict(color='#dc3545', width=2),
                marker=dict(size=8)
            ))
            fig_risk.add_hline(y=0.5, line_dash="dash", line_color="orange",
                              annotation_text="Risk Threshold (50%)")
            fig_risk.update_layout(
                title='30-Day Readmission Risk Trajectory',
                xaxis_title='Assessment Date',
                yaxis_title='Risk Probability',
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # B-line score and IVC trends
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bline = go.Figure()
                fig_bline.add_trace(go.Scatter(
                    x=patient_df['assessment_date'],
                    y=patient_df['b_line_score'],
                    mode='lines+markers',
                    name='B-line Score',
                    line=dict(color='#6c757d', width=2),
                    marker=dict(size=8)
                ))
                fig_bline.update_layout(
                    title='B-line Score Trend',
                    xaxis_title='Date',
                    yaxis_title='B-line Score',
                    height=350
                )
                st.plotly_chart(fig_bline, use_container_width=True)
            
            with col2:
                fig_ivc = go.Figure()
                fig_ivc.add_trace(go.Scatter(
                    x=patient_df['assessment_date'],
                    y=patient_df['ivc_collapsibility'],
                    mode='lines+markers',
                    name='IVC Collapsibility',
                    line=dict(color='#ffc107', width=2),
                    marker=dict(size=8)
                ))
                fig_ivc.update_layout(
                    title='IVC Collapsibility Trend',
                    xaxis_title='Date',
                    yaxis_title='IVC Collapsibility (%)',
                    height=350
                )
                st.plotly_chart(fig_ivc, use_container_width=True)
        else:
            st.info("Only one assessment available. Add more assessments to see trend analysis.")

def show_data_management_page(db_manager, data_generator):
    """Display data management page."""
    st.header("Data Management")
    
    # Database statistics
    st.subheader("Database Statistics")
    stats = db_manager.get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", stats.get('total_records', 0))
    with col2:
        st.metric("Database Size", f"{stats.get('db_size_mb', 0):.2f} MB")
    with col3:
        st.metric("Last Update", stats.get('last_update', 'Never'))
    
    # Data generation
    st.subheader("Generate Training Data")
    st.info("Generate additional synthetic training data to improve model performance.")
    
    n_samples = st.number_input("Number of samples to generate", min_value=100, max_value=10000, value=1000, step=100)
    
    if st.button("Generate Data"):
        with st.spinner("Generating synthetic data..."):
            synthetic_data = data_generator.generate_training_dataset(n_samples=n_samples)
            st.success(f"Generated {len(synthetic_data)} synthetic patient records")
            
            # Show sample of generated data
            st.subheader("Sample Generated Data")
            st.dataframe(synthetic_data.head(10))
    
    # Export functionality
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export All Assessments to CSV"):
            assessments = db_manager.get_all_assessments()
            if assessments:
                df = pd.DataFrame(assessments)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"all_assessments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No assessments to export")
    
    with col2:
        if st.button("📥 Export All Assessments to Excel"):
            assessments = db_manager.get_all_assessments()
            if assessments:
                df = pd.DataFrame(assessments)
                # Use openpyxl for Excel export
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Assessments', index=False)
                
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"all_assessments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No assessments to export")
    
    # Database management
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All Records", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                db_manager.clear_all_records()
                st.success("All records cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all records")
    
    with col2:
        if st.button("Reset Database", type="secondary"):
            if st.session_state.get('confirm_reset', False):
                db_manager.reset_database()
                st.success("Database reset!")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm database reset")

if __name__ == "__main__":
    main()
