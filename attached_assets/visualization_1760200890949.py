import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

class RiskVisualizer:
    """Visualization components for AHF risk assessment."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.color_scheme = {
            'low_risk': '#28a745',      # Green
            'moderate_risk': '#ffc107', # Yellow
            'high_risk': '#dc3545',     # Red
            'primary': '#007bff',       # Blue
            'secondary': '#6c757d'      # Gray
        }
    
    def plot_risk_gauge(self, risk_probability):
        """Create a risk gauge visualization."""
        # Determine risk level and color
        if risk_probability < 0.3:
            color = self.color_scheme['low_risk']
            risk_level = "Low Risk"
        elif risk_probability < 0.7:
            color = self.color_scheme['moderate_risk']
            risk_level = "Moderate Risk"
        else:
            color = self.color_scheme['high_risk']
            risk_level = "High Risk"
        
        # Create gauge chart with lighter colors for steps (opacity not supported)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"30-Day Readmission Risk<br><span style='color:{color};font-size:0.8em'>{risk_level}</span>"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(40, 167, 69, 0.3)'},  # Light green
                    {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.3)'},  # Light yellow
                    {'range': [70, 100], 'color': 'rgba(220, 53, 69, 0.3)'}  # Light red
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300, font={'size': 16})
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_feature_importance(self, importance_dict, patient_data):
        """Plot feature importance with patient-specific values."""
        if importance_dict is None:
            st.warning("Feature importance not available")
            return
        
        # Prepare data for plotting
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        # Map feature names to more readable labels
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
            'ckd': 'CKD',
            'afib': 'Atrial Fibrillation',
            'gender': 'Gender'
        }
        
        df['Feature_Label'] = df['Feature'].map(feature_labels).fillna(df['Feature'])
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            df, 
            x='Importance', 
            y='Feature_Label',
            orientation='h',
            title='Feature Importance in Risk Prediction',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title='Importance Score',
            yaxis_title='Clinical Features',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show patient-specific feature values
        st.subheader("Patient Feature Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Laboratory Values:**")
            st.write(f"• NT-proBNP: {patient_data.get('nt_probnp', 'N/A'):.0f} pg/mL")
            st.write(f"• Creatinine: {patient_data.get('creatinine', 'N/A'):.2f} mg/dL")
            
            st.markdown("**Demographics:**")
            st.write(f"• Age: {patient_data.get('age', 'N/A')} years")
            st.write(f"• Gender: {'Male' if patient_data.get('gender', 0) == 1 else 'Female'}")
            st.write(f"• Weight: {patient_data.get('weight', 'N/A'):.1f} kg")
        
        with col2:
            st.markdown("**Ultrasound Parameters:**")
            st.write(f"• B-line Score: {patient_data.get('b_line_score', 'N/A')}")
            st.write(f"• IVC Collapsibility: {patient_data.get('ivc_collapsibility', 'N/A'):.1f}%")
            
            st.markdown("**Clinical Parameters:**")
            st.write(f"• Ejection Fraction: {patient_data.get('ejection_fraction', 'N/A'):.1f}%")
            st.write(f"• Systolic BP: {patient_data.get('systolic_bp', 'N/A')} mmHg")
            st.write(f"• Heart Rate: {patient_data.get('heart_rate', 'N/A')} bpm")
    
    def plot_roc_curves(self, performance_metrics):
        """Plot ROC curves for both models."""
        fig = go.Figure()
        
        # Plot ROC curves for both models
        for model_name, metrics in performance_metrics.items():
            if 'true_labels' in metrics and 'probabilities' in metrics:
                fpr, tpr, _ = roc_curve(metrics['true_labels'], metrics['probabilities'])
                auc = metrics['auc']
                
                model_display_name = model_name.replace('_', ' ').title()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_display_name} (AUC = {auc:.3f})',
                    line=dict(width=2)
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves - Model Performance Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_confusion_matrices(self, performance_metrics):
        """Plot confusion matrices for both models."""
        col1, col2 = st.columns(2)
        
        for i, (model_name, metrics) in enumerate(performance_metrics.items()):
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                model_display_name = model_name.replace('_', ' ').title()
                
                # Create heatmap
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f'{model_display_name} - Confusion Matrix',
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Readmission', 'Readmission'],
                    y=['No Readmission', 'Readmission'],
                    color_continuous_scale='Blues'
                )
                
                if i == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
    
    def plot_probability_distribution(self, predictions_df):
        """Plot distribution of risk probabilities."""
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=predictions_df['risk_probability'],
            nbinsx=20,
            name='Risk Probability Distribution',
            marker_color=self.color_scheme['primary'],
            opacity=0.7
        ))
        
        # Add vertical lines for risk thresholds
        fig.add_vline(x=0.3, line_dash="dash", line_color=self.color_scheme['low_risk'],
                     annotation_text="Low Risk Threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color=self.color_scheme['high_risk'],
                     annotation_text="High Risk Threshold")
        
        fig.update_layout(
            title='Distribution of 30-Day Readmission Risk Probabilities',
            xaxis_title='Risk Probability',
            yaxis_title='Number of Patients',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_biomarker_trends(self, patient_history):
        """Plot biomarker trends over time for a patient."""
        if not patient_history:
            st.info("No historical data available for trend analysis")
            return
        
        df = pd.DataFrame(patient_history)
        df['assessment_date'] = pd.to_datetime(df['assessment_date'])
        
        # Create subplots
        fig = go.Figure()
        
        # NT-proBNP trend
        fig.add_trace(go.Scatter(
            x=df['assessment_date'],
            y=df['nt_probnp'],
            mode='lines+markers',
            name='NT-proBNP',
            line=dict(color=self.color_scheme['primary'], width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='NT-proBNP Trend Over Time',
            xaxis_title='Assessment Date',
            yaxis_title='NT-proBNP (pg/mL)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk probability trend
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=df['assessment_date'],
            y=df['ensemble_probability'],
            mode='lines+markers',
            name='Risk Probability',
            line=dict(color=self.color_scheme['high_risk'], width=2),
            marker=dict(size=8)
        ))
        
        # Add risk threshold lines
        fig2.add_hline(y=0.3, line_dash="dash", line_color=self.color_scheme['low_risk'],
                      annotation_text="Low Risk Threshold")
        fig2.add_hline(y=0.7, line_dash="dash", line_color=self.color_scheme['high_risk'],
                      annotation_text="High Risk Threshold")
        
        fig2.update_layout(
            title='30-Day Readmission Risk Trend',
            xaxis_title='Assessment Date',
            yaxis_title='Risk Probability',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def create_risk_summary_card(self, risk_prob, risk_level, model_name):
        """Create a summary card for risk assessment."""
        color = self.color_scheme['low_risk'] if risk_level == "Low Risk" else \
                self.color_scheme['moderate_risk'] if risk_level == "Moderate Risk" else \
                self.color_scheme['high_risk']
        
        st.markdown(f"""
        <div style="
            border: 2px solid {color};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: rgba(255,255,255,0.1);
        ">
            <h4 style="color: {color}; margin: 0;">{model_name}</h4>
            <h2 style="margin: 10px 0;">{risk_prob:.1%}</h2>
            <p style="color: {color}; font-weight: bold; margin: 0;">{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def plot_calibration_curve(self, y_true, y_prob, model_name):
        """Plot calibration curve for model reliability assessment."""
        from sklearn.calibration import calibration_curve
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, normalize=False
        )
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Model calibration
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name=f'{model_name} Calibration',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'{model_name} - Calibration Plot',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=500,
            height=400
        )
        
        return fig
